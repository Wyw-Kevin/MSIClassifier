#!/usr/bin/env python3
"""Rebuild the scRNA MSI/MMR analysis on the shared compressed66 classes."""

from __future__ import annotations

import json
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC


FINAL = Path(__file__).resolve().parents[1]
ROOT = FINAL.parent
OUT = FINAL / "data" / "scRNA" / "result" / "diagnostics" / "compressed66_unified_rebuild_20260908"
RANDOM_STATE = 2026


@dataclass(frozen=True)
class ClassContract:
    class_names: tuple[str, ...]
    class_to_group: dict[str, str]
    subtype_to_class: dict[str, str]
    cell_counts: dict[str, int]


def _split_members(value: object) -> list[str]:
    if pd.isna(value) or not str(value).strip():
        return []
    return [member.strip() for member in str(value).split(";") if member.strip()]


def load_class_contract(
    definitions_path: str | Path,
    counts_path: str | Path,
    min_cells: int = 51,
) -> ClassContract:
    definitions = pd.read_csv(definitions_path, dtype=str)
    counts = pd.read_csv(counts_path)

    required_definitions = {"feature", "feature_type", "group_class", "subclass", "member_subclasses"}
    required_counts = {"CellType", "n_cells"}
    if not required_definitions.issubset(definitions.columns):
        missing = sorted(required_definitions.difference(definitions.columns))
        raise ValueError(f"feature definitions missing columns: {missing}")
    if not required_counts.issubset(counts.columns):
        missing = sorted(required_counts.difference(counts.columns))
        raise ValueError(f"cell counts missing columns: {missing}")

    definitions["feature"] = definitions["feature"].str.strip()
    counts["CellType"] = counts["CellType"].astype(str).str.strip()
    if definitions["feature"].duplicated().any():
        duplicated = sorted(definitions.loc[definitions["feature"].duplicated(False), "feature"].unique())
        raise ValueError(f"duplicate feature definitions: {duplicated}")
    if counts["CellType"].duplicated().any():
        duplicated = sorted(counts.loc[counts["CellType"].duplicated(False), "CellType"].unique())
        raise ValueError(f"duplicate cell-count definitions: {duplicated}")

    subtype_to_class: dict[str, str] = {}
    for row in definitions.itertuples(index=False):
        feature = str(row.feature).strip()
        members = _split_members(row.member_subclasses)
        source_subtypes = members if members else [str(row.subclass).strip() or feature]
        for subtype in source_subtypes:
            if subtype in subtype_to_class:
                raise ValueError(
                    f"duplicate subtype member definition: {subtype} maps to both "
                    f"{subtype_to_class[subtype]} and {feature}"
                )
            subtype_to_class[subtype] = feature

    counts["n_cells"] = pd.to_numeric(counts["n_cells"], errors="raise").astype(int)
    retained = counts.loc[counts["n_cells"].ge(min_cells)].copy()
    class_names = tuple(retained["CellType"].tolist())
    definition_by_feature = definitions.set_index("feature")
    undefined = sorted(set(class_names).difference(definition_by_feature.index))
    if undefined:
        raise ValueError(f"retained classes missing feature definitions: {undefined}")
    if len(class_names) != 66:
        raise ValueError(f"expected 66 retained classes, found {len(class_names)}")

    class_to_group = {
        feature: str(definition_by_feature.loc[feature, "group_class"]).strip()
        for feature in class_names
    }
    cell_counts = dict(zip(retained["CellType"], retained["n_cells"].astype(int)))
    return ClassContract(
        class_names=class_names,
        class_to_group=class_to_group,
        subtype_to_class=subtype_to_class,
        cell_counts=cell_counts,
    )


def build_group_fraction_matrix(
    cells: pd.DataFrame,
    sample_labels: pd.DataFrame,
    contract: ClassContract,
) -> pd.DataFrame:
    required_cells = {"sampleID", "raw_subtype", "major_class"}
    required_labels = {"sampleID", "MMR"}
    if not required_cells.issubset(cells.columns):
        missing = sorted(required_cells.difference(cells.columns))
        raise ValueError(f"cell table missing columns: {missing}")
    if not required_labels.issubset(sample_labels.columns):
        missing = sorted(required_labels.difference(sample_labels.columns))
        raise ValueError(f"sample-label table missing columns: {missing}")

    labels = sample_labels.loc[:, ["sampleID", "MMR"]].copy()
    labels["sampleID"] = labels["sampleID"].astype(str)
    labels["MMR"] = labels["MMR"].astype(str)
    conflicts = labels.groupby("sampleID", sort=False)["MMR"].nunique(dropna=False)
    conflicts = conflicts[conflicts.gt(1)]
    if not conflicts.empty:
        raise ValueError(f"conflicting labels for samples: {sorted(conflicts.index.tolist())}")
    labels = labels.drop_duplicates("sampleID", keep="first").set_index("sampleID")

    working = cells.loc[:, ["sampleID", "raw_subtype", "major_class"]].copy()
    working["sampleID"] = working["sampleID"].astype(str)
    working["raw_subtype"] = working["raw_subtype"].astype("string")
    working["major_class"] = working["major_class"].astype("string")
    unknown_samples = sorted(set(working["sampleID"]).difference(labels.index))
    if unknown_samples:
        raise ValueError(f"cells lack sample labels: {unknown_samples}")

    working["mapped_class"] = working["raw_subtype"].map(contract.subtype_to_class)
    unknown_mask = working["raw_subtype"].notna() & working["mapped_class"].isna()
    if unknown_mask.any():
        unknown = sorted(working.loc[unknown_mask, "raw_subtype"].astype(str).unique())
        raise ValueError(f"unknown subtypes: {unknown}")

    retained = working[working["mapped_class"].isin(contract.class_names)].copy()
    retained["group_class"] = retained["mapped_class"].map(contract.class_to_group)
    if retained["group_class"].isna().any():
        missing = sorted(retained.loc[retained["group_class"].isna(), "mapped_class"].unique())
        raise ValueError(f"retained classes missing group definitions: {missing}")

    if retained.empty:
        matrix = pd.DataFrame(0.0, index=labels.index, columns=contract.class_names)
    else:
        numerators = retained.groupby(
            ["sampleID", "group_class", "mapped_class"], observed=True
        ).size()
        denominators = numerators.groupby(level=[0, 1]).transform("sum")
        fractions = (numerators / denominators).rename("fraction").reset_index()
        matrix = fractions.pivot(index="sampleID", columns="mapped_class", values="fraction")
        matrix = matrix.reindex(index=labels.index, columns=contract.class_names, fill_value=0.0)
        matrix = matrix.fillna(0.0).astype(float)

        represented_sums = fractions.groupby(["sampleID", "group_class"])["fraction"].sum()
        if not ((represented_sums - 1.0).abs() < 1e-12).all():
            raise ValueError("within-group fractions do not sum to one")

    matrix.index.name = "sampleID"
    matrix.insert(0, "MMR", labels.loc[matrix.index, "MMR"].to_numpy())
    return matrix


def _unique_sample_labels(frame: pd.DataFrame) -> pd.DataFrame:
    labels = frame.loc[:, ["sampleID", "MMR"]].copy()
    labels["sampleID"] = labels["sampleID"].astype(str)
    labels["MMR"] = labels["MMR"].astype(str)
    conflicts = labels.groupby("sampleID", sort=False)["MMR"].nunique(dropna=False)
    conflicts = conflicts[conflicts.gt(1)]
    if not conflicts.empty:
        raise ValueError(f"conflicting labels for samples: {sorted(conflicts.index.tolist())}")
    labels = labels.drop_duplicates("sampleID", keep="first")
    invalid = sorted(set(labels["MMR"]).difference({"MSI", "MSS"}))
    if invalid:
        raise ValueError(f"unsupported MMR labels: {invalid}")
    return labels.reset_index(drop=True)


def load_gse178341_cells(h5ad_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    import anndata as ad

    dataset = ad.read_h5ad(h5ad_path, backed="r")
    required = {"sample", "MMR", "cancer", "majorCluster", "cl295v11SubShort"}
    try:
        missing = sorted(required.difference(dataset.obs.columns))
        if missing:
            raise ValueError(f"GSE178341 obs missing columns: {missing}")
        obs = dataset.obs.loc[:, sorted(required)].copy()
    finally:
        dataset.file.close()

    tumor_mask = obs["cancer"].astype("string").str.lower().eq("tumor").fillna(False)
    tumor = obs.loc[tumor_mask].copy()
    labels = _unique_sample_labels(
        tumor.loc[:, ["sample", "MMR"]].rename(columns={"sample": "sampleID"})
    )
    nonepithelial_mask = tumor["majorCluster"].astype("string").ne("Epi").fillna(False)
    cells = tumor.loc[
        nonepithelial_mask, ["sample", "cl295v11SubShort", "majorCluster"]
    ].rename(
        columns={
            "sample": "sampleID",
            "cl295v11SubShort": "raw_subtype",
            "majorCluster": "major_class",
        }
    )
    cells["sampleID"] = cells["sampleID"].astype(str)
    return cells.reset_index(drop=True), labels


def load_gse132465_cells(
    migration_path: str | Path,
    processed_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    migration_columns = ["Sample", "Class", "major_prediction", "final_migrated_subtype"]
    migration = pd.read_csv(migration_path, usecols=migration_columns)
    processed = pd.read_csv(processed_path, usecols=["sampleID", "MMR"])
    labels = _unique_sample_labels(processed)

    tumor_mask = migration["Class"].astype("string").str.lower().eq("tumor").fillna(False)
    nonepithelial_mask = migration["major_prediction"].astype("string").ne("Epi").fillna(False)
    cells = migration.loc[
        tumor_mask & nonepithelial_mask,
        ["Sample", "final_migrated_subtype", "major_prediction"],
    ].rename(
        columns={
            "Sample": "sampleID",
            "final_migrated_subtype": "raw_subtype",
            "major_prediction": "major_class",
        }
    )
    cells["sampleID"] = cells["sampleID"].astype(str)
    return cells.reset_index(drop=True), labels


def attach_mlh1_features(
    matrix: pd.DataFrame,
    source_paths: list[str | Path],
) -> pd.DataFrame:
    if matrix.index.name != "sampleID" or "MMR" not in matrix.columns:
        raise ValueError("feature matrix must be indexed by sampleID and contain MMR")

    result = matrix.copy()
    result.index = result.index.astype(str)
    combined = pd.DataFrame(
        index=result.index,
        columns=["mean_MLH1_epithelial", "n_epithelial_cells_for_MLH1"],
        dtype=float,
    )
    for source_path in source_paths:
        header = pd.read_csv(source_path, nrows=0)
        required = {"sampleID", "mean_MLH1_epithelial"}
        missing = sorted(required.difference(header.columns))
        if missing:
            raise ValueError(f"MLH1 source missing columns: {missing}")
        usecols = ["sampleID", "mean_MLH1_epithelial"]
        if "n_epithelial_cells_for_MLH1" in header.columns:
            usecols.append("n_epithelial_cells_for_MLH1")
        if "MMR" in header.columns:
            usecols.append("MMR")
        source = pd.read_csv(source_path, usecols=usecols)
        source["sampleID"] = source["sampleID"].astype(str)
        if source["sampleID"].duplicated().any():
            duplicated = sorted(source.loc[source["sampleID"].duplicated(False), "sampleID"].unique())
            raise ValueError(f"duplicate sample IDs in MLH1 source: {duplicated}")
        source = source.set_index("sampleID")

        overlap = result.index.intersection(source.index)
        if "MMR" in source.columns:
            source_labels = source.loc[overlap, "MMR"].astype(str)
            expected_labels = result.loc[overlap, "MMR"].astype(str)
            conflicts = overlap[source_labels.to_numpy() != expected_labels.to_numpy()]
            if len(conflicts):
                raise ValueError(f"conflicting labels in MLH1 source: {sorted(conflicts.tolist())}")

        if "n_epithelial_cells_for_MLH1" not in source.columns:
            source["n_epithelial_cells_for_MLH1"] = np.nan
        values = source.reindex(result.index)[
            ["mean_MLH1_epithelial", "n_epithelial_cells_for_MLH1"]
        ].apply(pd.to_numeric, errors="coerce")
        combined = combined.combine_first(values)

    missing_samples = combined.index[combined["mean_MLH1_epithelial"].isna()].tolist()
    if missing_samples:
        raise ValueError(f"missing epithelial MLH1 values for samples: {missing_samples}")
    result["mean_MLH1_epithelial"] = combined["mean_MLH1_epithelial"].astype(float)
    result["n_epithelial_cells_for_MLH1"] = combined["n_epithelial_cells_for_MLH1"].astype(float)
    return result


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def screen_features(
    train: pd.DataFrame,
    candidate_features: list[str] | tuple[str, ...],
    class_to_group: dict[str, str] | None = None,
) -> pd.DataFrame:
    if "MMR" not in train.columns:
        raise ValueError("training matrix is missing MMR")
    invalid_labels = sorted(set(train["MMR"].astype(str)).difference({"MSI", "MSS"}))
    if invalid_labels:
        raise ValueError(f"unsupported MMR labels: {invalid_labels}")
    y = train["MMR"].astype(str).eq("MSI").to_numpy(dtype=int)
    if len(np.unique(y)) != 2:
        raise ValueError("feature screening requires both MSI and MSS samples")

    rows = []
    for feature in candidate_features:
        if feature not in train.columns:
            raise ValueError(f"training matrix is missing candidate feature: {feature}")
        values = pd.to_numeric(train[feature], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"candidate feature contains non-finite values: {feature}")
        msi_values = values[y == 1]
        mss_values = values[y == 0]
        auc_msi_high = float(roc_auc_score(y, values))
        delta = float(msi_values.mean() - mss_values.mean())
        if delta > 0:
            effect_direction = "MSI_up"
            direction_sign = 1
        elif delta < 0:
            effect_direction = "MSI_down"
            direction_sign = -1
        else:
            effect_direction = "no_difference"
            direction_sign = 0
        rows.append(
            {
                "feature": str(feature),
                "group_class": (class_to_group or {}).get(str(feature), ""),
                "n": int(len(values)),
                "MSI": int(np.sum(y == 1)),
                "MSS": int(np.sum(y == 0)),
                "MSI_mean": float(msi_values.mean()),
                "MSS_mean": float(mss_values.mean()),
                "delta": delta,
                "wilcox_p": float(mannwhitneyu(msi_values, mss_values, alternative="two-sided").pvalue),
                "auc_msi_high": auc_msi_high,
                "auc_best_direction": float(max(auc_msi_high, 1.0 - auc_msi_high)),
                "effect_direction": effect_direction,
                "direction_sign": direction_sign,
            }
        )
    stats = pd.DataFrame(rows)
    stats["bh_fdr"] = _benjamini_hochberg(stats["wilcox_p"].to_numpy())
    return stats.sort_values(
        ["auc_best_direction", "feature"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)


def select_auc_features(stats: pd.DataFrame, threshold: float = 0.65) -> list[str]:
    required = {"feature", "auc_best_direction"}
    if not required.issubset(stats.columns):
        raise ValueError(f"screening table missing columns: {sorted(required.difference(stats.columns))}")
    return stats.loc[stats["auc_best_direction"].gt(threshold), "feature"].astype(str).tolist()


def build_feature_sets(
    ranked_features: list[str],
    minimum_count: int = 10,
) -> dict[str, list[str]]:
    if len(ranked_features) < minimum_count:
        raise ValueError(
            f"only {len(ranked_features)} features passed screening; at least {minimum_count} are required"
        )
    return {
        f"top_{count}_plus_mlh1": list(ranked_features[:count])
        for count in range(minimum_count, len(ranked_features) + 1)
    }


def rank_train_only(metrics: pd.DataFrame, model_order: list[str]) -> pd.DataFrame:
    required = {
        "split",
        "dataset",
        "feature_set",
        "model",
        "n_subtype_features",
        "auc",
        "balanced_accuracy",
    }
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise ValueError(f"model metrics missing columns: {missing}")
    key_columns = ["feature_set", "model"]
    if "normalization" in metrics.columns:
        key_columns.append("normalization")

    oof = metrics.loc[
        metrics["split"].eq("train_5fold_oof") & metrics["dataset"].eq("GSE178341")
    ].copy()
    external = metrics.loc[
        metrics["split"].eq("external") & metrics["dataset"].eq("GSE132465")
    ].copy()
    if oof.empty:
        raise ValueError("no GSE178341 five-fold OOF rows available for ranking")

    metric_names = [
        "auc",
        "balanced_accuracy",
        "accuracy",
        "error_rate",
        "sensitivity",
        "specificity",
        "TP",
        "FN",
        "FP",
        "TN",
    ]
    available_oof = [column for column in metric_names if column in oof.columns]
    available_external = [column for column in metric_names if column in external.columns]
    ranked = oof[
        key_columns + ["n_subtype_features"] + available_oof
    ].rename(columns={column: f"train_oof_{column}" for column in available_oof})
    external_compact = external[key_columns + available_external].rename(
        columns={column: f"external_{column}" for column in available_external}
    )
    ranked = ranked.merge(external_compact, on=key_columns, how="left", validate="one_to_one")
    order_lookup = {name: position for position, name in enumerate(model_order)}
    ranked["model_order"] = ranked["model"].map(order_lookup).fillna(len(order_lookup)).astype(int)
    ranked = ranked.sort_values(
        [
            "train_oof_auc",
            "train_oof_balanced_accuracy",
            "n_subtype_features",
            "model_order",
            "feature_set",
        ],
        ascending=[False, False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked.insert(0, "train_rank", np.arange(1, len(ranked) + 1))
    return ranked


def _metrics_dict(y_true: np.ndarray, score: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    pred = np.asarray(pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    precision, _, f1, _ = precision_recall_fscore_support(
        y_true, pred, labels=[1], average="binary", zero_division=0
    )
    accuracy = float(accuracy_score(y_true, pred))
    return {
        "n": int(len(y_true)),
        "MSI": int(np.sum(y_true == 1)),
        "MSS": int(np.sum(y_true == 0)),
        "auc": float(roc_auc_score(y_true, score)) if len(np.unique(y_true)) == 2 else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "accuracy": accuracy,
        "error_rate": float(1.0 - accuracy),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else np.nan,
        "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
        "precision": float(precision),
        "f1": float(f1),
        "TP": int(tp),
        "FN": int(fn),
        "FP": int(fp),
        "TN": int(tn),
    }


def _score_and_predict(model: object, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        score = np.asarray(model.predict_proba(x)[:, classes.index(1)], dtype=float)
        return score, (score >= 0.5).astype(int), "probability_ge_0.5"
    if hasattr(model, "decision_function"):
        score = np.asarray(model.decision_function(x), dtype=float)
        if score.ndim > 1:
            score = score[:, list(model.classes_).index(1)]
        return score, np.asarray(model.predict(x), dtype=int), "model_default_margin"
    pred = np.asarray(model.predict(x), dtype=int)
    return pred.astype(float), pred, "model_default_class"


def _validate_model_matrix(frame: pd.DataFrame, features: list[str], name: str) -> np.ndarray:
    required = {"sampleID", "MMR", *features}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} matrix missing columns: {missing}")
    values = frame.loc[:, features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} model input contains non-finite values")
    return values


def evaluate_candidate(
    train: pd.DataFrame,
    external: pd.DataFrame,
    subtype_features: list[str],
    model_template: object,
    feature_set_name: str,
    model_name: str,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = list(subtype_features) + ["mean_MLH1_epithelial"]
    x_train = _validate_model_matrix(train, features, "training")
    x_external = _validate_model_matrix(external, features, "external")
    y_train = train["MMR"].astype(str).eq("MSI").to_numpy(dtype=int)
    y_external = external["MMR"].astype(str).eq("MSI").to_numpy(dtype=int)

    common = {
        "feature_set": feature_set_name,
        "n_subtype_features": int(len(subtype_features)),
        "n_model_features": int(len(features)),
        "normalization": "train_minmax",
        "model": model_name,
    }
    metrics_rows = []
    prediction_frames = []
    oof_score = np.zeros(len(train), dtype=float)
    oof_pred = np.zeros(len(train), dtype=int)
    oof_fold = np.zeros(len(train), dtype=int)
    score_rule = ""
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    for fold, (train_index, test_index) in enumerate(folds.split(x_train, y_train), start=1):
        scaler = MinMaxScaler()
        x_fold_train = scaler.fit_transform(x_train[train_index])
        x_fold_test = scaler.transform(x_train[test_index])
        model = clone(model_template)
        if hasattr(model, "random_state") and getattr(model, "random_state") is not None:
            model.set_params(random_state=2026)
        model.fit(x_fold_train, y_train[train_index])
        score, pred, score_rule = _score_and_predict(model, x_fold_test)
        oof_score[test_index] = score
        oof_pred[test_index] = pred
        oof_fold[test_index] = fold

    metrics_rows.append(
        {
            "split": "train_5fold_oof",
            "dataset": "GSE178341",
            "score_rule": score_rule,
            **common,
            **_metrics_dict(y_train, oof_score, oof_pred),
        }
    )
    prediction_frames.append(
        pd.DataFrame(
            {
                "split": "train_5fold_oof",
                "dataset": "GSE178341",
                **common,
                "sampleID": train["sampleID"].astype(str).to_numpy(),
                "MMR": train["MMR"].astype(str).to_numpy(),
                "y": y_train,
                "fold": oof_fold,
                "score": oof_score,
                "predicted_MSI": oof_pred,
            }
        )
    )

    final_scaler = MinMaxScaler()
    x_train_scaled = final_scaler.fit_transform(x_train)
    final_model = clone(model_template)
    final_model.fit(x_train_scaled, y_train)

    train_score, train_pred, train_rule = _score_and_predict(final_model, x_train_scaled)
    metrics_rows.append(
        {
            "split": "train_full_fit",
            "dataset": "GSE178341",
            "score_rule": train_rule,
            **common,
            **_metrics_dict(y_train, train_score, train_pred),
        }
    )
    prediction_frames.append(
        pd.DataFrame(
            {
                "split": "train_full_fit",
                "dataset": "GSE178341",
                **common,
                "sampleID": train["sampleID"].astype(str).to_numpy(),
                "MMR": train["MMR"].astype(str).to_numpy(),
                "y": y_train,
                "fold": 0,
                "score": train_score,
                "predicted_MSI": train_pred,
            }
        )
    )

    external_scaled = final_scaler.transform(x_external)
    external_score, external_pred, external_rule = _score_and_predict(final_model, external_scaled)
    metrics_rows.append(
        {
            "split": "external",
            "dataset": "GSE132465",
            "score_rule": external_rule,
            **common,
            **_metrics_dict(y_external, external_score, external_pred),
        }
    )
    prediction_frames.append(
        pd.DataFrame(
            {
                "split": "external",
                "dataset": "GSE132465",
                **common,
                "sampleID": external["sampleID"].astype(str).to_numpy(),
                "MMR": external["MMR"].astype(str).to_numpy(),
                "y": y_external,
                "fold": 0,
                "score": external_score,
                "predicted_MSI": external_pred,
            }
        )
    )

    if hasattr(final_model, "feature_importances_"):
        importance_values = np.asarray(final_model.feature_importances_, dtype=float)
        importance_type = "feature_importance"
    elif hasattr(final_model, "coef_"):
        importance_values = np.ravel(final_model.coef_).astype(float)
        importance_type = "coefficient"
    else:
        importance_values = np.repeat(np.nan, len(features))
        importance_type = "unavailable"
    importance = pd.DataFrame(
        {
            **common,
            "feature": features,
            "importance_or_coef": importance_values,
            "importance_type": importance_type,
        }
    )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions["predicted_MMR"] = np.where(predictions["predicted_MSI"].eq(1), "MSI", "MSS")
    predictions["error_type"] = np.select(
        [
            predictions["y"].eq(1) & predictions["predicted_MSI"].eq(1),
            predictions["y"].eq(1) & predictions["predicted_MSI"].eq(0),
            predictions["y"].eq(0) & predictions["predicted_MSI"].eq(1),
            predictions["y"].eq(0) & predictions["predicted_MSI"].eq(0),
        ],
        ["TP", "FN", "FP", "TN"],
        default="NA",
    )
    return pd.DataFrame(metrics_rows), predictions, importance


def run_model_grid(
    train: pd.DataFrame,
    external: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    models: dict[str, object],
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_tables = []
    prediction_tables = []
    importance_tables = []
    failure_rows = []
    for feature_set_name, subtype_features in feature_sets.items():
        for model_name, model_template in models.items():
            run_id = f"{feature_set_name}__train_minmax__{model_name}"
            try:
                metrics, predictions, importance = evaluate_candidate(
                    train=train,
                    external=external,
                    subtype_features=subtype_features,
                    model_template=model_template,
                    feature_set_name=feature_set_name,
                    model_name=model_name,
                    random_state=random_state,
                )
                metric_tables.append(metrics)
                prediction_tables.append(predictions)
                importance_tables.append(importance)
            except Exception as exc:
                failure_rows.append(
                    {
                        "run_id": run_id,
                        "feature_set": feature_set_name,
                        "model": model_name,
                        "error": repr(exc),
                    }
                )

    metrics = pd.concat(metric_tables, ignore_index=True) if metric_tables else pd.DataFrame()
    predictions = (
        pd.concat(prediction_tables, ignore_index=True) if prediction_tables else pd.DataFrame()
    )
    importance = (
        pd.concat(importance_tables, ignore_index=True) if importance_tables else pd.DataFrame()
    )
    failures = pd.DataFrame(
        failure_rows, columns=["run_id", "feature_set", "model", "error"]
    )
    return metrics, predictions, importance, failures


def make_models(random_state: int = RANDOM_STATE) -> dict[str, object]:
    return {
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
        "gradient_boosting_shallow": GradientBoostingClassifier(
            random_state=random_state,
            n_estimators=60,
            learning_rate=0.05,
            max_depth=2,
        ),
        "logistic_l2": LogisticRegression(
            max_iter=10000, solver="liblinear", random_state=random_state
        ),
        "logistic_l2_balanced": LogisticRegression(
            max_iter=10000,
            solver="liblinear",
            class_weight="balanced",
            random_state=random_state,
        ),
        "linear_svc": LinearSVC(max_iter=20000, dual=False, random_state=random_state),
        "linear_svc_balanced": LinearSVC(
            max_iter=20000,
            dual=False,
            class_weight="balanced",
            random_state=random_state,
        ),
        "ridge_classifier": RidgeClassifier(),
        "lda_shrinkage": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        "mlp_4_l2": MLPClassifier(
            hidden_layer_sizes=(4,), alpha=1.0, max_iter=5000, random_state=random_state
        ),
        "perceptron_balanced": Perceptron(
            class_weight="balanced", max_iter=10000, random_state=random_state
        ),
        "sgd_log_l2": SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=0.01,
            max_iter=10000,
            random_state=random_state,
        ),
    }


def _external_exploratory_ranking(
    metrics: pd.DataFrame,
    model_order: list[str],
) -> pd.DataFrame:
    external = metrics.loc[
        metrics["split"].eq("external") & metrics["dataset"].eq("GSE132465")
    ].copy()
    order_lookup = {name: position for position, name in enumerate(model_order)}
    external["model_order"] = external["model"].map(order_lookup).fillna(len(order_lookup)).astype(int)
    external = external.sort_values(
        ["auc", "balanced_accuracy", "n_subtype_features", "model_order", "feature_set"],
        ascending=[False, False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    external.insert(0, "external_exploratory_rank", np.arange(1, len(external) + 1))
    return external


def _feature_set_table(feature_sets: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for feature_set, features in feature_sets.items():
        for rank, feature in enumerate(features, start=1):
            rows.append(
                {
                    "feature_set": feature_set,
                    "n_subtype_features": len(features),
                    "within_set_rank": rank,
                    "feature": feature,
                }
            )
        rows.append(
            {
                "feature_set": feature_set,
                "n_subtype_features": len(features),
                "within_set_rank": len(features) + 1,
                "feature": "mean_MLH1_epithelial",
            }
        )
    return pd.DataFrame(rows)


def _contract_table(contract: ClassContract) -> pd.DataFrame:
    members_by_class: dict[str, list[str]] = {name: [] for name in contract.class_names}
    for subtype, class_name in contract.subtype_to_class.items():
        if class_name in members_by_class:
            members_by_class[class_name].append(subtype)
    return pd.DataFrame(
        {
            "class_name": contract.class_names,
            "group_class": [contract.class_to_group[name] for name in contract.class_names],
            "reference_n_cells": [contract.cell_counts[name] for name in contract.class_names],
            "source_subtypes": [";".join(sorted(members_by_class[name])) for name in contract.class_names],
        }
    )


def _old18_comparison(stats: pd.DataFrame, selected_features: list[str]) -> pd.DataFrame:
    old_features_path = FINAL / "scRNA_model" / "reference" / "selected_18_subtype_features.csv"
    old_stats_path = (
        FINAL
        / "model_data"
        / "single_cell"
        / "model_results"
        / "modle2"
        / "outputs_scRNA_gse178341_subtype_auc_screen_20260713"
        / "GSE178341_subtype_auc_stats.csv"
    )
    old_features = pd.read_csv(old_features_path)["feature"].astype(str).tolist()
    old_stats = pd.read_csv(old_stats_path)
    old_stats = old_stats.loc[old_stats["feature"].isin(old_features)].copy()
    old_stats = old_stats[
        ["feature", "train_n", "auc_best_direction", "effect_direction", "direction_sign"]
    ].rename(
        columns={
            "train_n": "archived_screen_n",
            "auc_best_direction": "archived_auc_best_direction",
            "effect_direction": "archived_effect_direction",
            "direction_sign": "archived_direction_sign",
        }
    )
    current = stats[
        ["feature", "n", "auc_best_direction", "effect_direction", "direction_sign"]
    ].rename(
        columns={
            "n": "unified66_screen_n",
            "auc_best_direction": "unified66_auc_best_direction",
            "effect_direction": "unified66_effect_direction",
            "direction_sign": "unified66_direction_sign",
        }
    )
    comparison = pd.DataFrame(
        {"feature": old_features, "archived_feature_rank": np.arange(1, len(old_features) + 1)}
    )
    comparison = comparison.merge(old_stats, on="feature", how="left", validate="one_to_one")
    comparison = comparison.merge(current, on="feature", how="left", validate="one_to_one")
    selected_rank = {feature: rank for rank, feature in enumerate(selected_features, start=1)}
    comparison["passes_unified66_auc_gt065"] = comparison["feature"].isin(selected_features)
    comparison["unified66_selected_rank"] = comparison["feature"].map(selected_rank)
    comparison["direction_same"] = (
        comparison["archived_direction_sign"] == comparison["unified66_direction_sign"]
    )
    return comparison


def _identity_mask(frame: pd.DataFrame, identity: pd.Series) -> pd.Series:
    return frame["feature_set"].eq(identity["feature_set"]) & frame["model"].eq(identity["model"])


def _json_scalar(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if pd.isna(value):
        return None
    return value


def main() -> None:
    definitions_path = FINAL / "data" / "scRNA" / "reference" / "hybrid_feature_definitions.csv"
    counts_path = (
        FINAL
        / "data"
        / "bulk"
        / "prepare"
        / "GSE178341_DECODE_reference"
        / "compressed66"
        / "reference"
        / "gse178341_decode_reference_tumor_full_protein_coding_celltype_counts.csv"
    )
    train_processed_path = FINAL / "data" / "scRNA" / "result" / "process" / "GSE178341_processed.csv"
    train_mlh1_preferred_path = (
        FINAL / "data" / "scRNA" / "input" / "gse178341_18_plus_mlh1_features.csv"
    )
    external_processed_path = FINAL / "data" / "scRNA" / "result" / "process" / "GSE132465_processed.csv"
    external_migration_path = (
        FINAL
        / "data"
        / "scRNA"
        / "result"
        / "process"
        / "GSE132465_tosica_migration_cells.csv"
    )

    OUT.mkdir(parents=True, exist_ok=True)
    contract = load_class_contract(definitions_path, counts_path, min_cells=51)
    train_cells, train_labels = load_gse178341_cells(ROOT / "GSE178341.h5ad")
    external_cells, external_labels = load_gse132465_cells(
        external_migration_path, external_processed_path
    )
    train_matrix = build_group_fraction_matrix(train_cells, train_labels, contract)
    external_matrix = build_group_fraction_matrix(external_cells, external_labels, contract)
    train_matrix = attach_mlh1_features(
        train_matrix, [train_mlh1_preferred_path, train_processed_path]
    )
    external_matrix = attach_mlh1_features(external_matrix, [external_processed_path])

    if len(train_matrix) != 64 or train_matrix["MMR"].value_counts().to_dict() != {"MSI": 35, "MSS": 29}:
        raise ValueError(
            f"unexpected GSE178341 cohort: n={len(train_matrix)}, "
            f"labels={train_matrix['MMR'].value_counts().to_dict()}"
        )
    if len(external_matrix) != 23 or external_matrix["MMR"].value_counts().to_dict() != {"MSS": 19, "MSI": 4}:
        raise ValueError(
            f"unexpected GSE132465 cohort: n={len(external_matrix)}, "
            f"labels={external_matrix['MMR'].value_counts().to_dict()}"
        )

    stats = screen_features(
        train_matrix.reset_index(),
        list(contract.class_names),
        class_to_group=contract.class_to_group,
    )
    selected_features = select_auc_features(stats, threshold=0.65)
    feature_sets = build_feature_sets(selected_features, minimum_count=10)
    models = make_models(RANDOM_STATE)
    train = train_matrix.reset_index()
    external = external_matrix.reset_index()
    metrics, predictions, importance, failures = run_model_grid(
        train, external, feature_sets, models, RANDOM_STATE
    )
    ranked = rank_train_only(metrics, list(models))
    external_ranked = _external_exploratory_ranking(metrics, list(models))
    primary = ranked.iloc[0]
    external_best = external_ranked.iloc[0]

    selected_metrics = pd.concat(
        [
            metrics.loc[_identity_mask(metrics, primary)].assign(
                selection="train_selected_primary"
            ),
            metrics.loc[_identity_mask(metrics, external_best)].assign(
                selection="external_best_exploratory"
            ),
        ],
        ignore_index=True,
    )

    _contract_table(contract).to_csv(OUT / "class_contract_66.csv", index=False)
    train.insert(0, "dataset", "GSE178341")
    external.insert(0, "dataset", "GSE132465")
    train.to_csv(OUT / "GSE178341_unified66_features.csv", index=False)
    external.to_csv(OUT / "GSE132465_unified66_features.csv", index=False)
    stats.to_csv(OUT / "GSE178341_unified66_univariate_auc.csv", index=False)
    stats.loc[stats["auc_best_direction"].gt(0.65)].to_csv(
        OUT / "GSE178341_unified66_auc_gt065.csv", index=False
    )
    _feature_set_table(feature_sets).to_csv(OUT / "feature_sets.csv", index=False)
    metrics.to_csv(OUT / "model_grid_metrics.csv", index=False)
    predictions.to_csv(OUT / "model_grid_sample_predictions.csv", index=False)
    importance.to_csv(OUT / "model_grid_feature_importance_or_coef.csv", index=False)
    failures.to_csv(OUT / "model_grid_failures.csv", index=False)
    ranked.to_csv(OUT / "model_grid_ranked_train_only.csv", index=False)
    selected_metrics.to_csv(OUT / "selected_model_metrics.csv", index=False)
    _old18_comparison(stats, selected_features).to_csv(
        OUT / "old18_overlap_and_direction.csv", index=False
    )

    identity_columns = ["feature_set", "model", "n_subtype_features"]
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "output_directory": str(OUT),
        "random_state": RANDOM_STATE,
        "candidate_contract": {
            "n_classes": len(contract.class_names),
            "minimum_reference_cells_inclusive": 51,
            "fraction_denominator": "retained mapped cells within each major class",
        },
        "screening": {
            "dataset": "GSE178341",
            "threshold_rule": "auc_best_direction > 0.65",
            "feature_selection_scope": "full development cohort before fixed-feature OOF",
            "n_selected": len(selected_features),
            "selected_features_ranked": selected_features,
        },
        "cohorts": {
            "GSE178341": train_matrix["MMR"].value_counts().to_dict(),
            "GSE132465": external_matrix["MMR"].value_counts().to_dict(),
        },
        "model_protocol": {
            "feature_counts": list(range(10, len(selected_features) + 1)),
            "always_included_feature": "mean_MLH1_epithelial",
            "normalization": "MinMaxScaler fit on each fold training split; final scaler fit on all GSE178341",
            "primary_selection": "training OOF AUC, OOF balanced accuracy, fewer subtype features, stable model order",
            "external_selection_status": "exploratory only; not used for primary selection",
        },
        "train_selected_primary": {
            column: _json_scalar(primary[column]) for column in identity_columns
        },
        "external_best_exploratory": {
            column: _json_scalar(external_best[column]) for column in identity_columns
        },
        "failed_model_runs": int(len(failures)),
        "archived_model_replaced": False,
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if not failures.empty:
        raise RuntimeError(f"{len(failures)} model-grid runs failed; see model_grid_failures.csv")

    print(f"output_directory={OUT}")
    print(f"candidate_classes={len(contract.class_names)}")
    print(f"train_samples={len(train_matrix)} labels={train_matrix['MMR'].value_counts().to_dict()}")
    print(
        f"external_samples={len(external_matrix)} "
        f"labels={external_matrix['MMR'].value_counts().to_dict()}"
    )
    print(f"auc_gt_0.65_features={len(selected_features)} {selected_features}")
    print(
        "train_selected_primary="
        f"{primary['feature_set']} {primary['model']} "
        f"oof_auc={primary['train_oof_auc']:.6f} "
        f"oof_ba={primary['train_oof_balanced_accuracy']:.6f}"
    )
    print(
        "external_best_exploratory="
        f"{external_best['feature_set']} {external_best['model']} "
        f"auc={external_best['auc']:.6f} ba={external_best['balanced_accuracy']:.6f}"
    )


if __name__ == "__main__":
    main()
