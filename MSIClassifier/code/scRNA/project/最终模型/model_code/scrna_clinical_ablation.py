#!/usr/bin/env python3
"""Leakage-isolated clinical-adjustment and ablation utilities for scRNA MSI models."""

from __future__ import annotations

# Packaging-only path resolution; archived originals retain exact source bytes.
from pathlib import Path as _MSIPackagePath
import sys as _MSIPackageSys
for _MSIPackageParent in _MSIPackagePath(__file__).resolve().parents:
    if (_MSIPackageParent / "project_paths.py").is_file():
        _MSIPackageSys.path.insert(0, str(_MSIPackageParent))
        break
from project_paths import project_root

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import run_scrna_metadata_adjusted_pipeline as core
from run_unified66_scrna_pipeline import _score_and_predict
from scrna_metadata_adjustment import (
    MetadataResidualizer,
    make_patient_grouped_folds,
    validate_covariate_names,
)


RANDOM_STATE = core.RANDOM_STATE
MLH1_FEATURE = core.MLH1_FEATURE


@dataclass(frozen=True)
class ClinicalTrack:
    continuous_covariates: tuple[str, ...]
    binary_covariates: tuple[str, ...]

    @property
    def covariates(self) -> tuple[str, ...]:
        return self.continuous_covariates + self.binary_covariates

    @property
    def adjusted(self) -> bool:
        return bool(self.covariates)


@dataclass
class CandidateSelection:
    feature_set: str
    subtype_features: tuple[str, ...]
    model_name: str
    inner_auc: float
    n_passing_screen: int
    failures: pd.DataFrame
    inner_candidates: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class NestedResult:
    metrics: pd.DataFrame
    scores: pd.DataFrame
    selections: pd.DataFrame
    failures: pd.DataFrame
    inner_candidates: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class OuterAblationResult:
    score: np.ndarray
    full_score: np.ndarray
    removed_features: tuple[str, ...]
    ablation_applied: bool


CLINICAL_TRACKS = {
    "expression_only": ClinicalTrack((), ()),
    "core_age_sex_side": ClinicalTrack(("age",), ("female", "right_sided")),
    "expanded_clinical6": ClinicalTrack(
        ("age",),
        ("female", "right_sided", "t_high", "node_positive", "grade_high"),
    ),
}
CONFIGURATIONS = ("ecology_plus_MLH1", "ecology_only", "MLH1_only")
LINEAGES = ("B", "Mast", "Myeloid", "Plasma", "Strom", "TNKILC")
CELL_COUNT_COVARIATES = {
    "n_tumor_cells",
    "n_epithelial_cells_for_MLH1",
    "log_tumor_cells",
    "log_mlh1_epithelial_cells",
}


def validate_track_contracts(
    tracks: dict[str, ClinicalTrack] = CLINICAL_TRACKS,
) -> None:
    for name, track in tracks.items():
        if len(set(track.covariates)) != len(track.covariates):
            raise ValueError(f"{name} contains duplicate covariates")
        forbidden = sorted(set(track.covariates).intersection(CELL_COUNT_COVARIATES))
        if forbidden:
            raise ValueError(f"{name} contains excluded cell-count covariates: {forbidden}")
        if track.adjusted:
            validate_covariate_names(list(track.covariates))


def validate_lineage_contract(
    candidate_features: list[str], class_to_group: dict[str, str]
) -> None:
    missing = sorted(set(candidate_features).difference(class_to_group))
    unsupported = sorted(
        {class_to_group[feature] for feature in candidate_features if feature in class_to_group}
        .difference(LINEAGES)
    )
    if missing or unsupported:
        raise ValueError(
            f"invalid lineage contract; missing={missing}, unsupported={unsupported}"
        )


def _additional_features(
    configuration: Literal["ecology_plus_MLH1", "ecology_only"],
) -> tuple[str, ...]:
    if configuration == "ecology_plus_MLH1":
        return (MLH1_FEATURE,)
    if configuration == "ecology_only":
        return ()
    raise ValueError(f"unsupported ecology configuration: {configuration}")


def _feature_sets(
    ranked_features: list[str],
    configuration: str,
    minimum_count: int,
    maximum_count: int,
) -> dict[str, list[str]]:
    if len(ranked_features) < minimum_count:
        raise ValueError(
            f"only {len(ranked_features)} candidates pass screening; "
            f"at least {minimum_count} are required"
        )
    suffix = "plus_mlh1" if configuration == "ecology_plus_MLH1" else "ecology_only"
    return {
        f"top_{count}_{suffix}": list(ranked_features[:count])
        for count in range(minimum_count, min(maximum_count, len(ranked_features)) + 1)
    }


def _screen_outer_training(
    train: pd.DataFrame,
    metadata: pd.DataFrame,
    candidate_features: list[str],
    inner_folds: list[tuple[np.ndarray, np.ndarray]],
    track: ClinicalTrack,
    threshold: float,
) -> list[str]:
    if track.adjusted:
        residuals, _ = core.cross_fitted_residuals(
            train,
            metadata,
            candidate_features,
            inner_folds,
            alpha=core.NUISANCE_ALPHA,
            continuous_covariates=track.continuous_covariates,
            binary_covariates=track.binary_covariates,
        )
        screen_frame = residuals.copy()
        screen_frame.insert(0, "MMR", train["MMR"].astype(str).to_numpy())
    else:
        screen_frame = train.loc[:, ["MMR"] + candidate_features].copy()
    statistics = core.screen_features(screen_frame, candidate_features)
    return statistics.loc[
        statistics["auc_best_direction"].gt(float(threshold)), "feature"
    ].astype(str).tolist()


def select_inner_candidate(
    train: pd.DataFrame,
    metadata: pd.DataFrame,
    candidate_features: list[str],
    models: dict[str, object],
    track: ClinicalTrack,
    configuration: Literal["ecology_plus_MLH1", "ecology_only"],
    threshold: float,
    minimum_count: int,
    maximum_count: int,
    inner_splits: int,
    random_state: int,
    normalization: str = "minmax",
) -> CandidateSelection:
    """Select a candidate using outer-training patients only."""

    inner_folds = make_patient_grouped_folds(
        train["MMR"].astype(str).eq("MSI").astype(int),
        metadata["patientID"].astype(str),
        n_splits=inner_splits,
        random_state=random_state,
    )
    ranked_features = _screen_outer_training(
        train,
        metadata,
        candidate_features,
        inner_folds,
        track,
        threshold,
    )
    feature_sets = _feature_sets(
        ranked_features, configuration, minimum_count, maximum_count
    )
    additional_features = _additional_features(configuration)

    # The development frame is reused as the grid's unused external argument so
    # outer-test data cannot enter selection through the existing grid API.
    if track.adjusted:
        metrics, _, _, failures = core.run_adjusted_model_grid(
            train=train,
            external=train,
            metadata_train=metadata,
            metadata_external=metadata,
            feature_sets=feature_sets,
            models=models,
            folds=inner_folds,
            random_state=random_state,
            alpha=core.NUISANCE_ALPHA,
            continuous_covariates=track.continuous_covariates,
            binary_covariates=track.binary_covariates,
            additional_features=additional_features,
            normalization=normalization,
        )
    else:
        metrics, _, failures = core.run_unadjusted_model_grid(
            train=train,
            external=train,
            metadata_train=metadata,
            metadata_external=metadata,
            feature_sets=feature_sets,
            models=models,
            folds=inner_folds,
            random_state=random_state,
            additional_features=additional_features,
            normalization=normalization,
        )
    if metrics.empty:
        raise ValueError(f"inner grid produced no metrics; failures={failures.to_dict('records')}")
    ranked = core.rank_adjusted_grid(metrics, list(models))
    winner = ranked.iloc[0]
    feature_set = str(winner["feature_set"])
    inner_candidates = ranked.rename(columns={"auc": "inner_auc"}).copy()
    inner_candidates["selected"] = inner_candidates["train_rank"].eq(1)
    return CandidateSelection(
        feature_set=feature_set,
        subtype_features=tuple(feature_sets[feature_set]),
        model_name=str(winner["model"]),
        inner_auc=float(winner["auc"]),
        n_passing_screen=len(ranked_features),
        inner_candidates=inner_candidates,
        failures=failures,
    )


def _fit_selected_and_score(
    train: pd.DataFrame,
    test: pd.DataFrame,
    metadata_train: pd.DataFrame,
    metadata_test: pd.DataFrame,
    model_features: list[str],
    model_template: object,
    track: ClinicalTrack,
    random_state: int,
    normalization: str = "minmax",
) -> np.ndarray:
    x_train = core._numeric_molecular(train, model_features, "outer training")
    x_test = core._numeric_molecular(test, model_features, "outer test")
    aligned_train = core._align_metadata(train, metadata_train, "outer training")
    aligned_test = core._align_metadata(test, metadata_test, "outer test")
    y_train = core._binary_labels(train, "outer training")
    if track.adjusted:
        residualizer = MetadataResidualizer(
            alpha=core.NUISANCE_ALPHA,
            continuous_covariates=track.continuous_covariates,
            binary_covariates=track.binary_covariates,
        )
        residualizer.fit(aligned_train, x_train)
        x_train = residualizer.transform(aligned_train, x_train)
        x_test = residualizer.transform(aligned_test, x_test)
    scaler_type = {"minmax": MinMaxScaler, "zscore": StandardScaler}[normalization]
    scaler = scaler_type().fit(x_train.to_numpy(dtype=float))
    model = clone(model_template)
    parameters = model.get_params(deep=False) if hasattr(model, "get_params") else {}
    if "random_state" in parameters and parameters["random_state"] is not None:
        model.set_params(random_state=int(random_state))
    model.fit(scaler.transform(x_train.to_numpy(dtype=float)), y_train)
    score, _, _ = _score_and_predict(
        model, scaler.transform(x_test.to_numpy(dtype=float))
    )
    return np.asarray(score, dtype=float)


def _nested_result(
    train: pd.DataFrame,
    metadata: pd.DataFrame,
    scores: np.ndarray,
    fold_numbers: np.ndarray,
    selections: list[dict[str, object]],
    inner_candidates: list[pd.DataFrame],
    failures: list[dict[str, object]],
    configuration: str,
) -> NestedResult:
    finite = np.isfinite(scores)
    metric_auc = (
        float(roc_auc_score(core._binary_labels(train, "development"), scores))
        if finite.all()
        else np.nan
    )
    metric_frame = pd.DataFrame(
        [
            {
                "split": "train_nested_grouped_oof",
                "dataset": "GSE178341",
                "configuration": configuration,
                "selection_protocol": "outer_isolated_inner_grouped_oof",
                "n": len(train),
                "MSI": int(train["MMR"].astype(str).eq("MSI").sum()),
                "MSS": int(train["MMR"].astype(str).eq("MSS").sum()),
                "auc": metric_auc,
            }
        ]
    )
    aligned = core._align_metadata(train, metadata, "development")
    score_frame = pd.DataFrame(
        {
            "split": "train_nested_grouped_oof",
            "dataset": "GSE178341",
            "configuration": configuration,
            "selection_protocol": "outer_isolated_inner_grouped_oof",
            "sampleID": train["sampleID"].astype(str).to_numpy(),
            "patientID": aligned["patientID"].astype(str).to_numpy(),
            "MMR": train["MMR"].astype(str).to_numpy(),
            "y": core._binary_labels(train, "development"),
            "fold": fold_numbers,
            "score": scores,
        }
    )
    return NestedResult(
        metrics=metric_frame,
        scores=score_frame,
        selections=pd.DataFrame(selections),
        inner_candidates=(
            pd.concat(inner_candidates, ignore_index=True)
            if inner_candidates
            else pd.DataFrame()
        ),
        failures=pd.DataFrame(
            failures,
            columns=["outer_fold", "configuration", "feature_set", "model", "error"],
        ),
    )


def run_nested_configuration(
    train: pd.DataFrame,
    metadata: pd.DataFrame,
    candidate_features: list[str],
    models: dict[str, object],
    outer_folds: list[tuple[np.ndarray, np.ndarray]],
    track: ClinicalTrack,
    configuration: Literal["ecology_plus_MLH1", "ecology_only"],
    threshold: float = 0.65,
    minimum_count: int = 10,
    maximum_count: int = 24,
    inner_splits: int = 5,
    random_state: int = RANDOM_STATE,
    normalization: str = "minmax",
) -> NestedResult:
    """Run fully nested selection without exposing outer-test data to selection."""

    validate_track_contracts()
    additional_features = _additional_features(configuration)
    aligned = core._align_metadata(train, metadata, "development")
    oof_score = np.full(len(train), np.nan, dtype=float)
    oof_fold = np.zeros(len(train), dtype=int)
    selections: list[dict[str, object]] = []
    inner_candidates: list[pd.DataFrame] = []
    failures: list[dict[str, object]] = []

    for outer_fold, (train_index, test_index) in enumerate(outer_folds, start=1):
        outer_train = train.iloc[train_index].reset_index(drop=True)
        outer_test = train.iloc[test_index].reset_index(drop=True)
        outer_train_meta = aligned.iloc[train_index].reset_index(drop=True)
        outer_test_meta = aligned.iloc[test_index].reset_index(drop=True)
        try:
            selection = select_inner_candidate(
                train=outer_train,
                metadata=outer_train_meta,
                candidate_features=candidate_features,
                models=models,
                track=track,
                configuration=configuration,
                threshold=threshold,
                minimum_count=minimum_count,
                maximum_count=maximum_count,
                inner_splits=inner_splits,
                random_state=2026,
                normalization=normalization,
            )
            model_features = list(selection.subtype_features) + list(additional_features)
            fold_score = _fit_selected_and_score(
                train=outer_train,
                test=outer_test,
                metadata_train=outer_train_meta,
                metadata_test=outer_test_meta,
                model_features=model_features,
                model_template=models[selection.model_name],
                track=track,
                random_state=2026,
                normalization=normalization,
            )
            if not selection.failures.empty:
                for failure in selection.failures.to_dict("records"):
                    failures.append(
                        {
                            "outer_fold": outer_fold,
                            "configuration": configuration,
                            **failure,
                        }
                    )
            oof_score[test_index] = fold_score
            oof_fold[test_index] = outer_fold
            inner_candidates.append(
                selection.inner_candidates.assign(
                    outer_fold=outer_fold,
                    configuration=configuration,
                )
            )
            selections.append(
                {
                    "outer_fold": outer_fold,
                    "configuration": configuration,
                    "n_outer_train": len(train_index),
                    "n_outer_test": len(test_index),
                    "n_passing_screen": selection.n_passing_screen,
                    "feature_set": selection.feature_set,
                    "n_subtype_features": len(selection.subtype_features),
                    "n_model_features": len(model_features),
                    "model": selection.model_name,
                    "inner_grouped_oof_auc": selection.inner_auc,
                    "selected_features": ";".join(selection.subtype_features),
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "outer_fold": outer_fold,
                    "configuration": configuration,
                    "feature_set": "",
                    "model": "",
                    "error": repr(exc),
                }
            )

    return _nested_result(
        train,
        aligned,
        oof_score,
        oof_fold,
        selections,
        inner_candidates,
        failures,
        configuration,
    )


def run_nested_mlh1_only(
    train: pd.DataFrame,
    metadata: pd.DataFrame,
    outer_folds: list[tuple[np.ndarray, np.ndarray]],
    track: ClinicalTrack,
    random_state: int = RANDOM_STATE,
) -> NestedResult:
    """Score MLH1 with fold-specific residualization and training-only orientation."""

    validate_track_contracts()
    aligned = core._align_metadata(train, metadata, "development")
    molecular = core._numeric_molecular(train, [MLH1_FEATURE], "development")
    y = core._binary_labels(train, "development")
    oof_score = np.full(len(train), np.nan, dtype=float)
    oof_fold = np.zeros(len(train), dtype=int)
    selections: list[dict[str, object]] = []
    inner_candidates: list[pd.DataFrame] = []
    failures: list[dict[str, object]] = []

    for outer_fold, (train_index, test_index) in enumerate(outer_folds, start=1):
        try:
            x_train = molecular.iloc[train_index].reset_index(drop=True)
            x_test = molecular.iloc[test_index].reset_index(drop=True)
            train_meta = aligned.iloc[train_index].reset_index(drop=True)
            test_meta = aligned.iloc[test_index].reset_index(drop=True)
            if track.adjusted:
                residualizer = MetadataResidualizer(
                    alpha=core.NUISANCE_ALPHA,
                    continuous_covariates=track.continuous_covariates,
                    binary_covariates=track.binary_covariates,
                )
                residualizer.fit(train_meta, x_train)
                x_train = residualizer.transform(train_meta, x_train)
                x_test = residualizer.transform(test_meta, x_test)
            train_values = x_train[MLH1_FEATURE].to_numpy(dtype=float)
            direction_sign = (
                1
                if roc_auc_score(y[train_index], train_values) >= 0.5
                else -1
            )
            oof_score[test_index] = (
                direction_sign * x_test[MLH1_FEATURE].to_numpy(dtype=float)
            )
            oof_fold[test_index] = outer_fold
            inner_candidates.append(
                pd.DataFrame(
                    [
                        {
                            "outer_fold": outer_fold,
                            "configuration": "MLH1_only",
                            "feature_set": "MLH1_only",
                            "n_subtype_features": 0,
                            "model": "directional_MLH1",
                            "inner_auc": np.nan,
                            "selected": True,
                        }
                    ]
                )
            )
            selections.append(
                {
                    "outer_fold": outer_fold,
                    "configuration": "MLH1_only",
                    "n_outer_train": len(train_index),
                    "n_outer_test": len(test_index),
                    "n_passing_screen": 0,
                    "feature_set": "MLH1_only",
                    "n_subtype_features": 0,
                    "n_model_features": 1,
                    "model": "directional_MLH1",
                    "inner_grouped_oof_auc": np.nan,
                    "selected_features": "",
                    "direction_sign": direction_sign,
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "outer_fold": outer_fold,
                    "configuration": "MLH1_only",
                    "feature_set": "MLH1_only",
                    "model": "directional_MLH1",
                    "error": repr(exc),
                }
            )

    return _nested_result(
        train,
        aligned,
        oof_score,
        oof_fold,
        selections,
        inner_candidates,
        failures,
        "MLH1_only",
    )


def ablate_selected_features(
    selected_features: list[str],
    class_to_group: dict[str, str],
    ablation: str,
) -> tuple[list[str], list[str]]:
    """Return the retained and removed subtype features for one ablation."""

    selected = list(selected_features)
    missing = sorted(set(selected).difference(class_to_group))
    if missing:
        raise ValueError(f"selected features are missing lineage assignments: {missing}")
    if ablation in {"full", "without_MLH1"}:
        return selected, []
    if ablation == "without_all_ecology":
        return [], selected
    if not ablation.startswith("without_"):
        raise ValueError(f"unsupported ablation: {ablation}")
    lineage = ablation.removeprefix("without_")
    if lineage not in LINEAGES:
        raise ValueError(f"unsupported lineage ablation: {lineage}")
    removed = [feature for feature in selected if class_to_group[feature] == lineage]
    removed_set = set(removed)
    retained = [feature for feature in selected if feature not in removed_set]
    return retained, removed


def run_conditional_outer_ablation(
    train: pd.DataFrame,
    test: pd.DataFrame,
    metadata_train: pd.DataFrame,
    metadata_test: pd.DataFrame,
    selected_features: list[str],
    selected_model_name: str,
    models: dict[str, object],
    class_to_group: dict[str, str],
    track: ClinicalTrack,
    ablation: str,
    random_state: int,
) -> OuterAblationResult:
    """Refit one fold after removing a component from its selected full model."""

    if selected_model_name not in models:
        raise ValueError(f"selected model is unavailable: {selected_model_name}")
    model_template = models[selected_model_name]
    full_features = list(selected_features) + [MLH1_FEATURE]
    full_score = _fit_selected_and_score(
        train,
        test,
        metadata_train,
        metadata_test,
        full_features,
        model_template,
        track,
        random_state,
    )
    retained_subtypes, removed_subtypes = ablate_selected_features(
        selected_features, class_to_group, ablation
    )
    if ablation == "full":
        return OuterAblationResult(full_score, full_score, (), False)
    if ablation == "without_MLH1":
        ablated_features = list(selected_features)
        removed_features = (MLH1_FEATURE,)
        applied = True
    elif ablation == "without_all_ecology":
        ablated_features = [MLH1_FEATURE]
        removed_features = tuple(removed_subtypes)
        applied = bool(removed_features)
    else:
        removed_features = tuple(removed_subtypes)
        applied = bool(removed_features)
        ablated_features = retained_subtypes + [MLH1_FEATURE]
    if not applied:
        return OuterAblationResult(full_score.copy(), full_score, removed_features, False)
    score = _fit_selected_and_score(
        train,
        test,
        metadata_train,
        metadata_test,
        ablated_features,
        model_template,
        track,
        random_state,
    )
    return OuterAblationResult(score, full_score, removed_features, True)


def run_conditional_ablation(
    train: pd.DataFrame,
    metadata: pd.DataFrame,
    outer_folds: list[tuple[np.ndarray, np.ndarray]],
    full_selections: pd.DataFrame,
    models: dict[str, object],
    class_to_group: dict[str, str],
    track: ClinicalTrack,
    random_state: int,
) -> pd.DataFrame:
    """Apply component removals to fold-specific full-model selections."""

    required = {"outer_fold", "selected_features", "model"}
    missing = sorted(required.difference(full_selections.columns))
    if missing:
        raise ValueError(f"full selections are missing columns: {missing}")
    aligned = core._align_metadata(train, metadata, "development")
    selection_by_fold = full_selections.set_index("outer_fold")
    ablations = (
        "full",
        "without_MLH1",
        "without_all_ecology",
    ) + tuple(f"without_{lineage}" for lineage in LINEAGES)
    rows: list[pd.DataFrame] = []
    for outer_fold, (train_index, test_index) in enumerate(outer_folds, start=1):
        if outer_fold not in selection_by_fold.index:
            raise ValueError(f"missing full-model selection for outer fold {outer_fold}")
        selection = selection_by_fold.loc[outer_fold]
        selected_features = [
            feature for feature in str(selection["selected_features"]).split(";") if feature
        ]
        outer_train = train.iloc[train_index].reset_index(drop=True)
        outer_test = train.iloc[test_index].reset_index(drop=True)
        outer_train_meta = aligned.iloc[train_index].reset_index(drop=True)
        outer_test_meta = aligned.iloc[test_index].reset_index(drop=True)
        for ablation in ablations:
            result = run_conditional_outer_ablation(
                train=outer_train,
                test=outer_test,
                metadata_train=outer_train_meta,
                metadata_test=outer_test_meta,
                selected_features=selected_features,
                selected_model_name=str(selection["model"]),
                models=models,
                class_to_group=class_to_group,
                track=track,
                ablation=ablation,
                random_state=2026,
            )
            rows.append(
                pd.DataFrame(
                    {
                        "split": "train_nested_grouped_oof",
                        "dataset": "GSE178341",
                        "selection_protocol": "outer_selected_full_model_conditional_ablation",
                        "ablation": ablation,
                        "sampleID": outer_test["sampleID"].astype(str).to_numpy(),
                        "patientID": outer_test_meta["patientID"].astype(str).to_numpy(),
                        "MMR": outer_test["MMR"].astype(str).to_numpy(),
                        "y": core._binary_labels(outer_test, "outer test"),
                        "fold": outer_fold,
                        "score": result.score,
                        "full_score": result.full_score,
                        "ablation_applied": result.ablation_applied,
                        "removed_features": ";".join(result.removed_features),
                        "n_removed_features": len(result.removed_features),
                        "selected_model": str(selection["model"]),
                        "selected_features": ";".join(selected_features),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def run_locked_leave_one_feature_out(
    train: pd.DataFrame,
    external: pd.DataFrame,
    metadata_train: pd.DataFrame,
    metadata_external: pd.DataFrame,
    selected_features: list[str],
    selected_model_name: str,
    models: dict[str, object],
    folds: list[tuple[np.ndarray, np.ndarray]],
    track: ClinicalTrack,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run exploratory leave-one-subtype-out evaluation on a locked panel."""

    if selected_model_name not in models:
        raise ValueError(f"selected model is unavailable: {selected_model_name}")
    variants = [("full", "", list(selected_features))]
    variants.extend(
        (
            f"without_{removed_feature}",
            removed_feature,
            [feature for feature in selected_features if feature != removed_feature],
        )
        for removed_feature in selected_features
    )
    metric_parts: list[pd.DataFrame] = []
    score_parts: list[pd.DataFrame] = []
    seed = RANDOM_STATE
    for ablation, removed_feature, retained in variants:
        feature_set = f"locked_{ablation}"
        if track.adjusted:
            metrics, scores, _ = core.evaluate_adjusted_candidate(
                train=train,
                external=external,
                metadata_train=metadata_train,
                metadata_external=metadata_external,
                subtype_features=retained,
                model_template=models[selected_model_name],
                feature_set_name=feature_set,
                model_name=selected_model_name,
                folds=folds,
                random_state=seed,
                alpha=core.NUISANCE_ALPHA,
                continuous_covariates=track.continuous_covariates,
                binary_covariates=track.binary_covariates,
                additional_features=(MLH1_FEATURE,),
            )
        else:
            metrics, scores = core.evaluate_unadjusted_candidate(
                train=train,
                external=external,
                metadata_train=metadata_train,
                metadata_external=metadata_external,
                model_features=retained + [MLH1_FEATURE],
                model_template=models[selected_model_name],
                model_name=selected_model_name,
                folds=folds,
                random_state=seed,
            )
            metrics.insert(2, "feature_set", feature_set)
            metrics.insert(3, "n_subtype_features", len(retained))
            metrics.insert(4, "n_model_features", len(retained) + 1)
            scores.insert(2, "feature_set", feature_set)
            scores.insert(3, "n_subtype_features", len(retained))
            scores.insert(4, "n_model_features", len(retained) + 1)
        for frame in (metrics, scores):
            frame["ablation"] = ablation
            frame["removed_feature"] = removed_feature
            frame["selection_protocol"] = "development_locked_exploratory"
            frame["random_state"] = seed
        metric_parts.append(metrics)
        score_parts.append(scores)
    return (
        pd.concat(metric_parts, ignore_index=True),
        pd.concat(score_parts, ignore_index=True),
    )


def _validated_auc_vectors(
    y: object,
    score: object,
    patient_ids: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(y, dtype=int)
    scores = np.asarray(score, dtype=float)
    groups = np.asarray(patient_ids).astype(str)
    if labels.ndim != 1 or scores.ndim != 1 or groups.ndim != 1:
        raise ValueError("labels, scores, and patient IDs must be one-dimensional")
    if not (len(labels) == len(scores) == len(groups)) or len(labels) == 0:
        raise ValueError("labels, scores, and patient IDs must be aligned and non-empty")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("AUROC inference requires both outcome classes")
    if not np.isfinite(scores).all():
        raise ValueError("AUROC inference requires finite scores")
    group_labels = pd.DataFrame({"patientID": groups, "y": labels})
    if group_labels.groupby("patientID", sort=False)["y"].nunique().gt(1).any():
        raise ValueError("a patient cannot contain both outcome classes")
    return labels, scores, groups


def stratified_patient_bootstrap_indices(
    y: object,
    patient_ids: object,
    n_bootstrap: int,
    random_state: int,
) -> list[np.ndarray]:
    """Draw patient clusters within outcome strata and return specimen indices."""

    labels = np.asarray(y, dtype=int)
    groups = np.asarray(patient_ids).astype(str)
    if labels.ndim != 1 or groups.ndim != 1 or len(labels) != len(groups) or not len(labels):
        raise ValueError("labels and patient IDs must be aligned non-empty vectors")
    if int(n_bootstrap) < 1:
        raise ValueError("n_bootstrap must be positive")
    group_frame = pd.DataFrame({"patientID": groups, "y": labels})
    if group_frame.groupby("patientID", sort=False)["y"].nunique().gt(1).any():
        raise ValueError("a patient cannot contain both outcome classes")
    group_labels = group_frame.drop_duplicates("patientID").set_index("patientID")["y"]
    if set(group_labels.unique()) != {0, 1}:
        raise ValueError("patient bootstrap requires both outcome classes")
    groups_by_class = {
        label: group_labels.index[group_labels.eq(label)].to_numpy(dtype=str)
        for label in (0, 1)
    }
    indices_by_group = {
        group: np.flatnonzero(groups == group) for group in group_labels.index
    }
    rng = np.random.default_rng(int(random_state))
    draws: list[np.ndarray] = []
    for _ in range(int(n_bootstrap)):
        sampled_groups = np.concatenate(
            [
                rng.choice(group_names, size=len(group_names), replace=True)
                for group_names in groups_by_class.values()
            ]
        )
        draws.append(
            np.concatenate([indices_by_group[group] for group in sampled_groups])
        )
    return draws


def paired_patient_bootstrap_auc_delta(
    y: object,
    full_score: object,
    ablated_score: object,
    patient_ids: object,
    n_bootstrap: int = 2000,
    random_state: int = RANDOM_STATE,
) -> dict[str, float | int]:
    """Estimate a paired full-minus-ablated AUROC difference by patient bootstrap."""

    summary, _ = paired_patient_bootstrap_auc_delta_with_replicates(
        y,
        full_score,
        ablated_score,
        patient_ids,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    return summary


def paired_patient_bootstrap_auc_delta_with_replicates(
    y: object,
    full_score: object,
    ablated_score: object,
    patient_ids: object,
    n_bootstrap: int = 2000,
    random_state: int = RANDOM_STATE,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Estimate paired AUROC difference and retain every patient-cluster draw."""

    labels, full, groups = _validated_auc_vectors(y, full_score, patient_ids)
    ablated = np.asarray(ablated_score, dtype=float)
    if ablated.ndim != 1 or len(ablated) != len(labels) or not np.isfinite(ablated).all():
        raise ValueError("ablated scores must be a finite aligned vector")
    full_auc = float(roc_auc_score(labels, full))
    ablated_auc = float(roc_auc_score(labels, ablated))
    replicate_rows = []
    for replicate, index in enumerate(
        stratified_patient_bootstrap_indices(
            labels, groups, n_bootstrap, random_state
        ),
        start=1,
    ):
        replicate_full_auc = float(roc_auc_score(labels[index], full[index]))
        replicate_comparator_auc = float(
            roc_auc_score(labels[index], ablated[index])
        )
        replicate_rows.append(
            {
                "replicate": replicate,
                "full_auc": replicate_full_auc,
                "comparator_auc": replicate_comparator_auc,
                "delta_auc": replicate_full_auc - replicate_comparator_auc,
            }
        )
    replicates = pd.DataFrame(
        replicate_rows,
        columns=["replicate", "full_auc", "comparator_auc", "delta_auc"],
    )
    bootstrap_delta = replicates["delta_auc"].to_numpy(dtype=float)
    lower, upper = np.quantile(bootstrap_delta, [0.025, 0.975])
    summary = {
        "full_auc": full_auc,
        "ablated_auc": ablated_auc,
        "delta_auc": full_auc - ablated_auc,
        "delta_ci_lower": float(lower),
        "delta_ci_upper": float(upper),
        "bootstrap_success": int(len(bootstrap_delta)),
        "n": int(len(labels)),
        "n_patients": int(pd.Series(groups).nunique()),
    }
    return summary, replicates


def patient_average_scores(
    y: object,
    score: object,
    patient_ids: object,
) -> pd.DataFrame:
    """Average specimen scores within patient after validating patient labels."""

    labels, scores, groups = _validated_auc_vectors(y, score, patient_ids)
    frame = pd.DataFrame({"patientID": groups, "y": labels, "score": scores})
    return (
        frame.groupby("patientID", sort=False, as_index=False)
        .agg(y=("y", "first"), score=("score", "mean"))
    )


def score_external_mlh1_only(
    train: pd.DataFrame,
    external: pd.DataFrame,
    metadata_train: pd.DataFrame,
    metadata_external: pd.DataFrame,
    track: ClinicalTrack,
) -> tuple[np.ndarray, int]:
    """Score external MLH1 after fitting adjustment and orientation on development."""

    x_train = core._numeric_molecular(train, [MLH1_FEATURE], "development")
    x_external = core._numeric_molecular(external, [MLH1_FEATURE], "external")
    aligned_train = core._align_metadata(train, metadata_train, "development")
    aligned_external = core._align_metadata(external, metadata_external, "external")
    y_train = core._binary_labels(train, "development")
    if track.adjusted:
        residualizer = MetadataResidualizer(
            alpha=core.NUISANCE_ALPHA,
            continuous_covariates=track.continuous_covariates,
            binary_covariates=track.binary_covariates,
        )
        residualizer.fit(aligned_train, x_train)
        x_train = residualizer.transform(aligned_train, x_train)
        x_external = residualizer.transform(aligned_external, x_external)
    development_values = x_train[MLH1_FEATURE].to_numpy(dtype=float)
    direction_sign = 1 if roc_auc_score(y_train, development_values) >= 0.5 else -1
    external_score = direction_sign * x_external[MLH1_FEATURE].to_numpy(dtype=float)
    return external_score, direction_sign
