#!/usr/bin/env python3
"""Run the metadata-adjusted scRNA MSI/MMR sensitivity analysis."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import joblib
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from run_unified66_scrna_pipeline import _score_and_predict, make_models, screen_features
from scrna_metadata_adjustment import (
    ADJUSTMENT_COVARIATES,
    BINARY_COVARIATES,
    CONTINUOUS_COVARIATES,
    MetadataResidualizer,
    make_patient_grouped_folds,
    overlap_weights,
    parse_gse132465_metadata,
    parse_gse178341_metadata,
    standardized_mean_difference,
    weighted_auc,
)


FINAL = Path(__file__).resolve().parents[1]
ROOT = FINAL.parent
SOURCE_DIR = FINAL / "data" / "scRNA" / "result" / "diagnostics" / "unmerged_raw_subtypes_min50_20260908"
OUT = (
    FINAL
    / "data"
    / "scRNA"
    / "result"
    / "diagnostics"
    / "metadata_adjusted_clinical6_unmerged71"
)
RANDOM_STATE = 2026
NUISANCE_ALPHA = 1.0
MLH1_FEATURE = "mean_MLH1_epithelial"
FEATURE_THRESHOLD = 0.65
MINIMUM_FEATURE_COUNT = 10
MAXIMUM_FEATURE_COUNT = 24
N_SPLITS = 5
N_BOOTSTRAP = 2000
TRAIN_FEATURES_PATH = SOURCE_DIR / "GSE178341_unmerged71_features.csv"
EXTERNAL_FEATURES_PATH = SOURCE_DIR / "GSE132465_unmerged71_features.csv"
CLASS_CONTRACT_PATH = SOURCE_DIR / "class_contract_71_unmerged.csv"
CURRENT_PREDICTIONS_PATH = SOURCE_DIR / "model_grid_sample_predictions.csv"
GSE178341_METADATA_PATH = (
    FINAL
    / "data"
    / "scRNA"
    / "reference"
    / "metadata_sources"
    / "GSE178341_crc10x_full_c295v4_submit_metatables.csv.gz"
)
GSE132465_METADATA_PATH = (
    FINAL
    / "data"
    / "scRNA"
    / "reference"
    / "metadata_sources"
    / "41588_2020_636_MOESM3_ESM.xlsx"
)
GSE132465_MAJOR_COUNTS_PATH = (
    FINAL / "data" / "scRNA" / "result" / "process" / "GSE132465_tosica_major_counts_by_sample.csv"
)
MANUSCRIPT_PATH = FINAL / "report" / "article" / "MMR多转录组分类器_第二版.docx"
CURRENT_SHARED18 = (
    "cTNI14",
    "cTNI18",
    "cS18",
    "cTNI13",
    "cTNI16",
    "cTNI05",
    "cTNI07",
    "cS27",
    "cTNI04",
    "cS17",
    "cM01",
    "cP1",
    "cM05",
    "cM06",
    "cTNI08",
    "cTNI19",
    "cTNI10",
    "cM04",
)
CLINICAL_BASELINE_COVARIATES = (
    "age",
    "female",
    "right_sided",
    "t_high",
    "node_positive",
    "grade_high",
)
ADJUSTMENT_SET_SPECS = {
    "age_sex": (("age",), ("female",)),
    "demographics_stage_grade": (
        ("age",),
        ("female", "t_high", "node_positive", "grade_high"),
    ),
    "all_clinical": (CONTINUOUS_COVARIATES, BINARY_COVARIATES),
}
REQUIRED_OUTPUT_TABLES = (
    "metadata_GSE178341.csv",
    "metadata_GSE132465.csv",
    "metadata_association_with_MSI.csv",
    "covariate_balance_overlap_weighting.csv",
    "current_model_grouped_and_weighted_AUROC.csv",
    "current_model_grouped_sample_scores.csv",
    "adjusted_univariate_AUROC_71.csv",
    "adjusted_AUROC_gt065_features.csv",
    "adjusted_feature_sets.csv",
    "adjusted_fixed_screen_model_grid_AUROC.csv",
    "adjusted_fixed_screen_sample_scores.csv",
    "adjusted_nested_OOF_AUROC.csv",
    "adjusted_nested_fold_selections.csv",
    "adjusted_selected_model_AUROC.csv",
    "adjusted_selected_model_sample_scores.csv",
    "locked_shared18_adjustment_set_AUROC.csv",
    "locked_shared18_adjustment_set_sample_scores.csv",
    "shared18_adjusted_overlap_and_direction.csv",
    "clinical_only_baseline_AUROC.csv",
)
THRESHOLD_DERIVED_COLUMNS = {
    "balanced_accuracy",
    "accuracy",
    "error_rate",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "TP",
    "FN",
    "FP",
    "TN",
    "predicted_MSI",
    "predicted_MMR",
    "error_type",
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def attach_external_tumor_cell_counts(
    metadata: pd.DataFrame,
    major_counts: pd.DataFrame,
) -> pd.DataFrame:
    """Attach total tumor-cell counts from the archived per-sample major counts."""

    required = {"Class", "Sample", "n_cells"}
    missing_columns = sorted(required.difference(major_counts.columns))
    if missing_columns:
        raise ValueError(f"external major counts are missing columns: {missing_columns}")
    if "sampleID" not in metadata.columns:
        raise ValueError("external metadata is missing sampleID")
    tumor = major_counts.loc[
        major_counts["Class"].astype(str).str.strip().str.lower().eq("tumor")
    ].copy()
    tumor["n_cells"] = pd.to_numeric(tumor["n_cells"], errors="coerce")
    if tumor["n_cells"].isna().any() or tumor["n_cells"].lt(0).any():
        raise ValueError("external tumor-cell counts must be finite and non-negative")
    counts = tumor.groupby(tumor["Sample"].astype(str))["n_cells"].sum()
    result = metadata.copy()
    result["n_tumor_cells"] = result["sampleID"].astype(str).map(counts)
    if result["n_tumor_cells"].isna().any():
        missing = result.loc[result["n_tumor_cells"].isna(), "sampleID"].astype(str).tolist()
        raise ValueError(f"missing external tumor-cell counts: {missing}")
    result["n_tumor_cells"] = result["n_tumor_cells"].astype(float)
    return result


def stratified_bootstrap_auc(
    y: pd.Series | np.ndarray,
    score: pd.Series | np.ndarray,
    groups: pd.Series | np.ndarray | None = None,
    n_bootstrap: int = 2000,
    random_state: int = RANDOM_STATE,
) -> tuple[float, float, float, int]:
    """Return AUROC and a deterministic stratified percentile confidence interval."""

    labels = np.asarray(y, dtype=int)
    scores = np.asarray(score, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("bootstrap labels and scores must be aligned vectors")
    if set(np.unique(labels)) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError("bootstrap AUROC requires two classes and finite scores")
    if int(n_bootstrap) < 1:
        raise ValueError("n_bootstrap must be positive")

    rng = np.random.default_rng(int(random_state))
    bootstrap_values: list[float] = []
    if groups is None:
        units_by_class = {
            label: np.flatnonzero(labels == label) for label in (0, 1)
        }

        def sampled_indices() -> np.ndarray:
            return np.concatenate(
                [
                    rng.choice(indices, size=len(indices), replace=True)
                    for indices in units_by_class.values()
                ]
            )

    else:
        group_values = np.asarray(groups).astype(str)
        if group_values.ndim != 1 or len(group_values) != len(labels):
            raise ValueError("bootstrap groups must align with labels")
        group_frame = pd.DataFrame({"group": group_values, "y": labels})
        if group_frame.groupby("group")["y"].nunique().gt(1).any():
            raise ValueError("a bootstrap group cannot contain both classes")
        group_labels = group_frame.drop_duplicates("group").set_index("group")["y"]
        units_by_class = {
            label: group_labels.index[group_labels.eq(label)].to_numpy(dtype=str)
            for label in (0, 1)
        }
        index_by_group = {
            group: np.flatnonzero(group_values == group) for group in group_labels.index
        }

        def sampled_indices() -> np.ndarray:
            sampled_groups = np.concatenate(
                [
                    rng.choice(group_names, size=len(group_names), replace=True)
                    for group_names in units_by_class.values()
                ]
            )
            return np.concatenate([index_by_group[group] for group in sampled_groups])

    for _ in range(int(n_bootstrap)):
        indices = sampled_indices()
        bootstrap_values.append(float(roc_auc_score(labels[indices], scores[indices])))
    lower, upper = np.quantile(bootstrap_values, [0.025, 0.975])
    return (
        float(roc_auc_score(labels, scores)),
        float(lower),
        float(upper),
        len(bootstrap_values),
    )


def write_output_bundle(
    output_dir: str | Path,
    tables: dict[str, pd.DataFrame],
    manifest: dict[str, object],
) -> None:
    """Validate and write the AUROC-only sensitivity-analysis result bundle."""

    missing_tables = sorted(set(REQUIRED_OUTPUT_TABLES).difference(tables))
    if missing_tables:
        raise ValueError(f"required output tables are missing: {missing_tables}")
    for name in REQUIRED_OUTPUT_TABLES:
        forbidden = sorted(THRESHOLD_DERIVED_COLUMNS.intersection(tables[name].columns))
        if forbidden:
            raise ValueError(f"{name} contains threshold-derived columns: {forbidden}")

    stamped_manifest = json.loads(json.dumps(manifest, default=str))
    for source_name, source_record in stamped_manifest.get("source_files", {}).items():
        if "path" not in source_record:
            raise ValueError(f"manifest source {source_name} is missing path")
        source_path = Path(source_record["path"])
        if not source_path.is_file():
            raise ValueError(f"manifest source does not exist: {source_path}")
        source_record["sha256"] = _sha256(source_path)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_OUTPUT_TABLES:
        tables[name].to_csv(destination / name, index=False)
    with (destination / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(stamped_manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _align_metadata(features: pd.DataFrame, metadata: pd.DataFrame, name: str) -> pd.DataFrame:
    if "sampleID" not in features.columns or "sampleID" not in metadata.columns:
        raise ValueError(f"{name} features and metadata require sampleID")
    if features["sampleID"].duplicated().any() or metadata["sampleID"].duplicated().any():
        raise ValueError(f"{name} sampleID values must be unique")
    indexed = metadata.set_index(metadata["sampleID"].astype(str), drop=False)
    sample_ids = features["sampleID"].astype(str)
    missing = sorted(set(sample_ids).difference(indexed.index))
    extra = sorted(set(indexed.index).difference(sample_ids))
    if missing or extra:
        raise ValueError(f"{name} feature/metadata samples differ; missing={missing}, extra={extra}")
    return indexed.loc[sample_ids].reset_index(drop=True)


def _numeric_molecular(frame: pd.DataFrame, features: list[str], name: str) -> pd.DataFrame:
    missing = sorted(set(features).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} molecular features are missing: {missing}")
    result = frame.loc[:, features].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError(f"{name} molecular features contain non-finite values")
    return result


def _binary_labels(frame: pd.DataFrame, name: str) -> np.ndarray:
    if "MMR" not in frame.columns:
        raise ValueError(f"{name} is missing MMR")
    labels = frame["MMR"].astype(str)
    invalid = sorted(set(labels).difference({"MSI", "MSS"}))
    if invalid:
        raise ValueError(f"{name} contains unsupported MMR labels: {invalid}")
    y = labels.eq("MSI").to_numpy(dtype=int)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError(f"{name} requires both MMR classes")
    return y


def cross_fitted_residuals(
    train: pd.DataFrame,
    metadata: pd.DataFrame,
    molecular_features: list[str],
    folds: list[tuple[np.ndarray, np.ndarray]],
    alpha: float = NUISANCE_ALPHA,
    continuous_covariates: tuple[str, ...] = CONTINUOUS_COVARIATES,
    binary_covariates: tuple[str, ...] = BINARY_COVARIATES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate one held-out nuisance residual for every development specimen."""

    aligned_metadata = _align_metadata(train, metadata, "development")
    molecular = _numeric_molecular(train, molecular_features, "development")
    residuals = pd.DataFrame(
        np.nan, index=np.arange(len(train)), columns=molecular_features, dtype=float
    )
    fold_number = np.zeros(len(train), dtype=int)
    for fold, (train_index, test_index) in enumerate(folds, start=1):
        if np.any(fold_number[test_index] != 0):
            raise ValueError("a development sample appears in more than one held-out fold")
        residualizer = MetadataResidualizer(
            alpha=alpha,
            continuous_covariates=continuous_covariates,
            binary_covariates=binary_covariates,
        )
        residualizer.fit(
            aligned_metadata.iloc[train_index].reset_index(drop=True),
            molecular.iloc[train_index].reset_index(drop=True),
        )
        held_out = residualizer.transform(
            aligned_metadata.iloc[test_index].reset_index(drop=True),
            molecular.iloc[test_index].reset_index(drop=True),
        )
        residuals.iloc[test_index] = held_out.to_numpy(dtype=float)
        fold_number[test_index] = fold
    if np.any(fold_number == 0) or not np.isfinite(residuals.to_numpy()).all():
        raise ValueError("cross-fitted residuals do not cover every development sample")
    assignment = pd.DataFrame(
        {
            "sampleID": train["sampleID"].astype(str).to_numpy(),
            "patientID": aligned_metadata["patientID"].astype(str).to_numpy(),
            "fold": fold_number,
        }
    )
    return residuals, assignment


def _auc_row(
    split: str,
    dataset: str,
    y: np.ndarray,
    score: np.ndarray,
    feature_set_name: str,
    subtype_features: list[str],
    additional_features: tuple[str, ...],
    model_name: str,
    score_rule: str,
) -> dict[str, object]:
    return {
        "split": split,
        "dataset": dataset,
        "feature_set": feature_set_name,
        "n_subtype_features": len(subtype_features),
        "n_model_features": len(subtype_features) + len(additional_features),
        "model": model_name,
        "normalization": "fold_residualization_then_train_minmax",
        "score_rule": score_rule,
        "n": len(y),
        "MSI": int(np.sum(y == 1)),
        "MSS": int(np.sum(y == 0)),
        "auc": float(roc_auc_score(y, score)),
    }


def _prediction_frame(
    split: str,
    dataset: str,
    source: pd.DataFrame,
    metadata: pd.DataFrame,
    y: np.ndarray,
    score: np.ndarray,
    fold: np.ndarray,
    feature_set_name: str,
    subtype_features: list[str],
    additional_features: tuple[str, ...],
    model_name: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": split,
            "dataset": dataset,
            "feature_set": feature_set_name,
            "n_subtype_features": len(subtype_features),
            "n_model_features": len(subtype_features) + len(additional_features),
            "model": model_name,
            "sampleID": source["sampleID"].astype(str).to_numpy(),
            "patientID": metadata["patientID"].astype(str).to_numpy(),
            "MMR": source["MMR"].astype(str).to_numpy(),
            "y": y,
            "fold": fold,
            "score": score,
        }
    )


def evaluate_unadjusted_candidate(
    train: pd.DataFrame,
    external: pd.DataFrame,
    metadata_train: pd.DataFrame,
    metadata_external: pd.DataFrame,
    model_features: list[str],
    model_template: object,
    model_name: str,
    folds: list[tuple[np.ndarray, np.ndarray]],
    random_state: int = RANDOM_STATE,
    normalization: str = "minmax",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate an expression-only reference with patient-grouped folds."""

    scaler_types = {"minmax": MinMaxScaler, "zscore": StandardScaler}
    if normalization not in scaler_types:
        raise ValueError(f"Unknown normalization: {normalization}")
    scaler_type = scaler_types[normalization]
    aligned_train_meta = _align_metadata(train, metadata_train, "development")
    aligned_external_meta = _align_metadata(external, metadata_external, "external")
    x_train = _numeric_molecular(train, model_features, "development")
    x_external = _numeric_molecular(external, model_features, "external")
    y_train = _binary_labels(train, "development")
    y_external = _binary_labels(external, "external")
    oof_score = np.full(len(train), np.nan, dtype=float)
    oof_fold = np.zeros(len(train), dtype=int)

    for fold, (train_index, test_index) in enumerate(folds, start=1):
        scaler = scaler_type().fit(x_train.iloc[train_index].to_numpy(dtype=float))
        model = clone(model_template)
        parameters = model.get_params(deep=False) if hasattr(model, "get_params") else {}
        if "random_state" in parameters and parameters["random_state"] is not None:
            model.set_params(random_state=2026)
        model.fit(
            scaler.transform(x_train.iloc[train_index].to_numpy(dtype=float)),
            y_train[train_index],
        )
        score, _, _ = _score_and_predict(
            model, scaler.transform(x_train.iloc[test_index].to_numpy(dtype=float))
        )
        oof_score[test_index] = score
        oof_fold[test_index] = fold
    if not np.isfinite(oof_score).all() or np.any(oof_fold == 0):
        raise ValueError("unadjusted grouped OOF did not score every development sample")

    scaler = scaler_type().fit(x_train.to_numpy(dtype=float))
    final_model = clone(model_template)
    final_model.fit(scaler.transform(x_train.to_numpy(dtype=float)), y_train)
    full_score, _, _ = _score_and_predict(
        final_model, scaler.transform(x_train.to_numpy(dtype=float))
    )
    external_score, _, _ = _score_and_predict(
        final_model, scaler.transform(x_external.to_numpy(dtype=float))
    )
    metrics = pd.DataFrame(
        [
            {
                "split": "train_grouped_oof",
                "dataset": "GSE178341",
                "model": model_name,
                "normalization": f"fold_train_{normalization}",
                "n": len(y_train),
                "MSI": int(y_train.sum()),
                "MSS": int(len(y_train) - y_train.sum()),
                "auc": float(roc_auc_score(y_train, oof_score)),
            },
            {
                "split": "train_full_fit",
                "dataset": "GSE178341",
                "model": model_name,
                "normalization": f"full_train_{normalization}",
                "n": len(y_train),
                "MSI": int(y_train.sum()),
                "MSS": int(len(y_train) - y_train.sum()),
                "auc": float(roc_auc_score(y_train, full_score)),
            },
            {
                "split": "external",
                "dataset": "GSE132465",
                "model": model_name,
                "normalization": f"development_train_{normalization}",
                "n": len(y_external),
                "MSI": int(y_external.sum()),
                "MSS": int(len(y_external) - y_external.sum()),
                "auc": float(roc_auc_score(y_external, external_score)),
            },
        ]
    )
    predictions = pd.concat(
        [
            pd.DataFrame(
                {
                    "split": "train_grouped_oof",
                    "dataset": "GSE178341",
                    "model": model_name,
                    "sampleID": train["sampleID"].astype(str).to_numpy(),
                    "patientID": aligned_train_meta["patientID"].astype(str).to_numpy(),
                    "MMR": train["MMR"].astype(str).to_numpy(),
                    "y": y_train,
                    "fold": oof_fold,
                    "score": oof_score,
                }
            ),
            pd.DataFrame(
                {
                    "split": "train_full_fit",
                    "dataset": "GSE178341",
                    "model": model_name,
                    "sampleID": train["sampleID"].astype(str).to_numpy(),
                    "patientID": aligned_train_meta["patientID"].astype(str).to_numpy(),
                    "MMR": train["MMR"].astype(str).to_numpy(),
                    "y": y_train,
                    "fold": 0,
                    "score": full_score,
                }
            ),
            pd.DataFrame(
                {
                    "split": "external",
                    "dataset": "GSE132465",
                    "model": model_name,
                    "sampleID": external["sampleID"].astype(str).to_numpy(),
                    "patientID": aligned_external_meta["patientID"].astype(str).to_numpy(),
                    "MMR": external["MMR"].astype(str).to_numpy(),
                    "y": y_external,
                    "fold": 0,
                    "score": external_score,
                }
            ),
        ],
        ignore_index=True,
    )
    return metrics, predictions


def evaluate_clinical_baseline(
    train: pd.DataFrame,
    external: pd.DataFrame,
    metadata_train: pd.DataFrame,
    metadata_external: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    covariates: list[str],
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate a harmonized clinical-only logistic baseline."""

    aligned_train = _align_metadata(train, metadata_train, "development")
    aligned_external = _align_metadata(external, metadata_external, "external")
    missing = sorted(set(covariates).difference(aligned_train.columns))
    if missing:
        raise ValueError(f"clinical baseline covariates are missing: {missing}")
    y_train = _binary_labels(train, "development")
    y_external = _binary_labels(external, "external")
    continuous = [name for name in covariates if name in {"age"}]
    binary = [name for name in covariates if name not in continuous]

    def make_pipeline(seed: int) -> Pipeline:
        preprocess = ColumnTransformer(
            [
                (
                    "continuous",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="median")),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    continuous,
                ),
                (
                    "binary",
                    SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                    binary,
                ),
            ],
            remainder="drop",
        )
        return Pipeline(
            [
                ("preprocess", preprocess),
                ("model", LogisticRegression(C=1.0, max_iter=10000, random_state=seed)),
            ]
        )

    oof_score = np.full(len(train), np.nan, dtype=float)
    oof_fold = np.zeros(len(train), dtype=int)
    for fold, (train_index, test_index) in enumerate(folds, start=1):
        model = make_pipeline(2026)
        model.fit(aligned_train.iloc[train_index][covariates], y_train[train_index])
        oof_score[test_index] = model.predict_proba(
            aligned_train.iloc[test_index][covariates]
        )[:, 1]
        oof_fold[test_index] = fold
    if not np.isfinite(oof_score).all() or np.any(oof_fold == 0):
        raise ValueError("clinical grouped OOF did not score every development sample")

    final_model = make_pipeline(int(random_state))
    final_model.fit(aligned_train[covariates], y_train)
    external_score = final_model.predict_proba(aligned_external[covariates])[:, 1]
    metrics = pd.DataFrame(
        [
            {
                "split": "train_grouped_oof",
                "dataset": "GSE178341",
                "model": "clinical_logistic_l2",
                "covariates": ";".join(covariates),
                "n": len(y_train),
                "MSI": int(y_train.sum()),
                "MSS": int(len(y_train) - y_train.sum()),
                "auc": float(roc_auc_score(y_train, oof_score)),
            },
            {
                "split": "external",
                "dataset": "GSE132465",
                "model": "clinical_logistic_l2",
                "covariates": ";".join(covariates),
                "n": len(y_external),
                "MSI": int(y_external.sum()),
                "MSS": int(len(y_external) - y_external.sum()),
                "auc": float(roc_auc_score(y_external, external_score)),
            },
        ]
    )
    scores = pd.concat(
        [
            pd.DataFrame(
                {
                    "split": "train_grouped_oof",
                    "dataset": "GSE178341",
                    "sampleID": train["sampleID"].astype(str).to_numpy(),
                    "patientID": aligned_train["patientID"].astype(str).to_numpy(),
                    "MMR": train["MMR"].astype(str).to_numpy(),
                    "y": y_train,
                    "fold": oof_fold,
                    "score": oof_score,
                }
            ),
            pd.DataFrame(
                {
                    "split": "external",
                    "dataset": "GSE132465",
                    "sampleID": external["sampleID"].astype(str).to_numpy(),
                    "patientID": aligned_external["patientID"].astype(str).to_numpy(),
                    "MMR": external["MMR"].astype(str).to_numpy(),
                    "y": y_external,
                    "fold": 0,
                    "score": external_score,
                }
            ),
        ],
        ignore_index=True,
    )
    return metrics, scores


def evaluate_adjusted_candidate(
    train: pd.DataFrame,
    external: pd.DataFrame,
    metadata_train: pd.DataFrame,
    metadata_external: pd.DataFrame,
    subtype_features: list[str],
    model_template: object,
    feature_set_name: str,
    model_name: str,
    folds: list[tuple[np.ndarray, np.ndarray]],
    random_state: int = RANDOM_STATE,
    alpha: float = NUISANCE_ALPHA,
    continuous_covariates: tuple[str, ...] = CONTINUOUS_COVARIATES,
    binary_covariates: tuple[str, ...] = BINARY_COVARIATES,
    additional_features: tuple[str, ...] = (MLH1_FEATURE,),
    normalization: str = "minmax",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate one candidate with fold-specific nuisance adjustment and scaling."""

    scaler_type = {"minmax": MinMaxScaler, "zscore": StandardScaler}[normalization]
    model_features = list(subtype_features) + list(additional_features)
    if not model_features:
        raise ValueError("at least one molecular feature is required")
    aligned_train_meta = _align_metadata(train, metadata_train, "development")
    aligned_external_meta = _align_metadata(external, metadata_external, "external")
    x_train = _numeric_molecular(train, model_features, "development")
    x_external = _numeric_molecular(external, model_features, "external")
    y_train = _binary_labels(train, "development")
    y_external = _binary_labels(external, "external")

    oof_score = np.full(len(train), np.nan, dtype=float)
    oof_fold = np.zeros(len(train), dtype=int)
    score_rule = ""
    for fold, (train_index, test_index) in enumerate(folds, start=1):
        residualizer = MetadataResidualizer(
            alpha=alpha,
            continuous_covariates=continuous_covariates,
            binary_covariates=binary_covariates,
        )
        fold_train = x_train.iloc[train_index].reset_index(drop=True)
        fold_test = x_train.iloc[test_index].reset_index(drop=True)
        fold_train_meta = aligned_train_meta.iloc[train_index].reset_index(drop=True)
        fold_test_meta = aligned_train_meta.iloc[test_index].reset_index(drop=True)
        residualizer.fit(fold_train_meta, fold_train)
        residual_train = residualizer.transform(fold_train_meta, fold_train)
        residual_test = residualizer.transform(fold_test_meta, fold_test)
        scaler = scaler_type().fit(residual_train.to_numpy(dtype=float))
        model = clone(model_template)
        parameters = model.get_params(deep=False) if hasattr(model, "get_params") else {}
        if "random_state" in parameters and parameters["random_state"] is not None:
            model.set_params(random_state=2026)
        model.fit(
            scaler.transform(residual_train.to_numpy(dtype=float)),
            y_train[train_index],
        )
        score, _, score_rule = _score_and_predict(
            model, scaler.transform(residual_test.to_numpy(dtype=float))
        )
        oof_score[test_index] = score
        oof_fold[test_index] = fold
    if not np.isfinite(oof_score).all() or np.any(oof_fold == 0):
        raise ValueError("adjusted OOF evaluation did not score every development sample")

    residualizer = MetadataResidualizer(
        alpha=alpha,
        continuous_covariates=continuous_covariates,
        binary_covariates=binary_covariates,
    )
    residualizer.fit(aligned_train_meta, x_train)
    residual_train = residualizer.transform(aligned_train_meta, x_train)
    residual_external = residualizer.transform(aligned_external_meta, x_external)
    scaler = scaler_type().fit(residual_train.to_numpy(dtype=float))
    final_model = clone(model_template)
    final_model.fit(scaler.transform(residual_train.to_numpy(dtype=float)), y_train)
    full_score, _, full_rule = _score_and_predict(
        final_model, scaler.transform(residual_train.to_numpy(dtype=float))
    )
    external_score, _, external_rule = _score_and_predict(
        final_model, scaler.transform(residual_external.to_numpy(dtype=float))
    )

    metrics = pd.DataFrame(
        [
            _auc_row(
                "train_grouped_oof", "GSE178341", y_train, oof_score,
                feature_set_name, subtype_features, additional_features, model_name, score_rule,
            ),
            _auc_row(
                "train_full_fit", "GSE178341", y_train, full_score,
                feature_set_name, subtype_features, additional_features, model_name, full_rule,
            ),
            _auc_row(
                "external", "GSE132465", y_external, external_score,
                feature_set_name, subtype_features, additional_features, model_name, external_rule,
            ),
        ]
    )
    predictions = pd.concat(
        [
            _prediction_frame(
                "train_grouped_oof", "GSE178341", train, aligned_train_meta,
                y_train, oof_score, oof_fold, feature_set_name, subtype_features,
                additional_features, model_name,
            ),
            _prediction_frame(
                "train_full_fit", "GSE178341", train, aligned_train_meta,
                y_train, full_score, np.zeros(len(train), dtype=int),
                feature_set_name, subtype_features, additional_features, model_name,
            ),
            _prediction_frame(
                "external", "GSE132465", external, aligned_external_meta,
                y_external, external_score, np.zeros(len(external), dtype=int),
                feature_set_name, subtype_features, additional_features, model_name,
            ),
        ],
        ignore_index=True,
    )

    if hasattr(final_model, "feature_importances_"):
        importance_values = np.asarray(final_model.feature_importances_, dtype=float)
        importance_type = "feature_importance"
    elif hasattr(final_model, "coef_"):
        importance_values = np.ravel(final_model.coef_).astype(float)
        importance_type = "coefficient"
    else:
        importance_values = np.full(len(model_features), np.nan)
        importance_type = "unavailable"
    importance = pd.DataFrame(
        {
            "feature_set": feature_set_name,
            "model": model_name,
            "feature": model_features,
            "importance_or_coef": importance_values,
            "importance_type": importance_type,
        }
    )
    return metrics, predictions, importance


def build_adjusted_feature_sets(
    ranked_features: list[str],
    minimum_count: int = 10,
    maximum_count: int = 24,
) -> dict[str, list[str]]:
    if len(ranked_features) < minimum_count:
        raise ValueError(
            f"only {len(ranked_features)} adjusted candidates are available; at least {minimum_count} required"
        )
    upper = min(int(maximum_count), len(ranked_features))
    return {
        f"top_{count}_plus_mlh1": list(ranked_features[:count])
        for count in range(int(minimum_count), upper + 1)
    }


def run_adjusted_model_grid(
    train: pd.DataFrame,
    external: pd.DataFrame,
    metadata_train: pd.DataFrame,
    metadata_external: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    models: dict[str, object],
    folds: list[tuple[np.ndarray, np.ndarray]],
    random_state: int = RANDOM_STATE,
    alpha: float = NUISANCE_ALPHA,
    continuous_covariates: tuple[str, ...] = CONTINUOUS_COVARIATES,
    binary_covariates: tuple[str, ...] = BINARY_COVARIATES,
    additional_features: tuple[str, ...] = (MLH1_FEATURE,),
    normalization: str = "minmax",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics_parts = []
    prediction_parts = []
    importance_parts = []
    failures = []
    for feature_set_name, subtype_features in feature_sets.items():
        for model_name, model_template in models.items():
            try:
                metrics, predictions, importance = evaluate_adjusted_candidate(
                    train,
                    external,
                    metadata_train,
                    metadata_external,
                    subtype_features,
                    model_template,
                    feature_set_name,
                    model_name,
                    folds,
                    random_state,
                    alpha,
                    continuous_covariates,
                    binary_covariates,
                    additional_features,
                    normalization,
                )
                metrics_parts.append(metrics)
                prediction_parts.append(predictions)
                importance_parts.append(importance)
            except Exception as exc:
                failures.append(
                    {
                        "feature_set": feature_set_name,
                        "model": model_name,
                        "error": repr(exc),
                    }
                )
    metrics = pd.concat(metrics_parts, ignore_index=True) if metrics_parts else pd.DataFrame()
    predictions = (
        pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()
    )
    importance = (
        pd.concat(importance_parts, ignore_index=True) if importance_parts else pd.DataFrame()
    )
    failure_frame = pd.DataFrame(failures, columns=["feature_set", "model", "error"])
    return metrics, predictions, importance, failure_frame


def run_unadjusted_model_grid(
    train: pd.DataFrame,
    external: pd.DataFrame,
    metadata_train: pd.DataFrame,
    metadata_external: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    models: dict[str, object],
    folds: list[tuple[np.ndarray, np.ndarray]],
    random_state: int = RANDOM_STATE,
    additional_features: tuple[str, ...] = (MLH1_FEATURE,),
    normalization: str = "minmax",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the expression-only comparator on the same grouped folds and grid."""

    metric_parts = []
    prediction_parts = []
    failures = []
    for feature_set_name, subtype_features in feature_sets.items():
        for model_name, model_template in models.items():
            try:
                metrics, predictions = evaluate_unadjusted_candidate(
                    train,
                    external,
                    metadata_train,
                    metadata_external,
                    list(subtype_features) + list(additional_features),
                    model_template,
                    model_name,
                    folds,
                    random_state,
                    normalization,
                )
                metrics.insert(2, "feature_set", feature_set_name)
                metrics.insert(3, "n_subtype_features", len(subtype_features))
                metrics.insert(
                    4,
                    "n_model_features",
                    len(subtype_features) + len(additional_features),
                )
                predictions.insert(2, "feature_set", feature_set_name)
                predictions.insert(3, "n_subtype_features", len(subtype_features))
                predictions.insert(
                    4,
                    "n_model_features",
                    len(subtype_features) + len(additional_features),
                )
                metric_parts.append(metrics)
                prediction_parts.append(predictions)
            except Exception as exc:
                failures.append(
                    {
                        "feature_set": feature_set_name,
                        "model": model_name,
                        "error": repr(exc),
                    }
                )
    metrics = pd.concat(metric_parts, ignore_index=True) if metric_parts else pd.DataFrame()
    predictions = (
        pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()
    )
    failure_frame = pd.DataFrame(failures, columns=["feature_set", "model", "error"])
    return metrics, predictions, failure_frame


def run_nested_adjusted_oof(
    train: pd.DataFrame,
    metadata: pd.DataFrame,
    candidate_features: list[str],
    models: dict[str, object],
    outer_folds: list[tuple[np.ndarray, np.ndarray]],
    threshold: float = 0.65,
    minimum_count: int = 10,
    maximum_count: int = 24,
    inner_splits: int = 5,
    random_state: int = RANDOM_STATE,
    alpha: float = NUISANCE_ALPHA,
    continuous_covariates: tuple[str, ...] = CONTINUOUS_COVARIATES,
    binary_covariates: tuple[str, ...] = BINARY_COVARIATES,
    additional_features: tuple[str, ...] = (MLH1_FEATURE,),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Estimate the complete adjusted selection pipeline with nested grouped OOF."""

    aligned_metadata = _align_metadata(train, metadata, "development")
    model_features = list(candidate_features) + list(additional_features)
    molecular = _numeric_molecular(train, model_features, "development")
    y = _binary_labels(train, "development")
    oof_score = np.full(len(train), np.nan, dtype=float)
    oof_fold = np.zeros(len(train), dtype=int)
    selected_model = np.full(len(train), "", dtype=object)
    selected_feature_set = np.full(len(train), "", dtype=object)
    selections: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for outer_fold, (outer_train_index, outer_test_index) in enumerate(outer_folds, start=1):
        outer_train = train.iloc[outer_train_index].reset_index(drop=True)
        outer_test = train.iloc[outer_test_index].reset_index(drop=True)
        outer_train_meta = aligned_metadata.iloc[outer_train_index].reset_index(drop=True)
        outer_test_meta = aligned_metadata.iloc[outer_test_index].reset_index(drop=True)

        inner_folds = make_patient_grouped_folds(
            outer_train["MMR"].astype(str).eq("MSI").astype(int),
            outer_train_meta["patientID"].astype(str),
            n_splits=inner_splits,
            random_state=2026,
        )
        screening_residuals, _ = cross_fitted_residuals(
            outer_train,
            outer_train_meta,
            model_features,
            inner_folds,
            alpha=alpha,
            continuous_covariates=continuous_covariates,
            binary_covariates=binary_covariates,
        )
        screening_frame = screening_residuals.loc[:, candidate_features].copy()
        screening_frame.insert(0, "MMR", outer_train["MMR"].astype(str).to_numpy())
        screening_stats = screen_features(screening_frame, candidate_features)
        ranked_features = screening_stats.loc[
            screening_stats["auc_best_direction"].gt(float(threshold)), "feature"
        ].astype(str).tolist()

        try:
            feature_sets = build_adjusted_feature_sets(
                ranked_features,
                minimum_count=minimum_count,
                maximum_count=maximum_count,
            )
            grid_metrics, grid_predictions, _, grid_failures = run_adjusted_model_grid(
                outer_train,
                outer_test,
                outer_train_meta,
                outer_test_meta,
                feature_sets,
                models,
                inner_folds,
                random_state=2026,
                alpha=alpha,
                continuous_covariates=continuous_covariates,
                binary_covariates=binary_covariates,
                additional_features=additional_features,
            )
            if not grid_failures.empty:
                for row in grid_failures.to_dict("records"):
                    failures.append({"outer_fold": outer_fold, **row})
            ranked_grid = rank_adjusted_grid(grid_metrics, list(models))
            winner = ranked_grid.iloc[0]
            winner_set = str(winner["feature_set"])
            winner_model = str(winner["model"])
            winner_features = feature_sets[winner_set]
            held_out = grid_predictions.loc[
                grid_predictions["split"].eq("external")
                & grid_predictions["feature_set"].eq(winner_set)
                & grid_predictions["model"].eq(winner_model),
                ["sampleID", "score"],
            ]
            score_by_sample = held_out.set_index("sampleID")["score"]
            outer_scores = outer_test["sampleID"].astype(str).map(score_by_sample)
            if outer_scores.isna().any():
                raise ValueError("selected nested candidate did not score every outer test sample")
        except Exception as exc:
            failures.append(
                {
                    "outer_fold": outer_fold,
                    "feature_set": "",
                    "model": "",
                    "error": repr(exc),
                }
            )
            continue

        oof_score[outer_test_index] = outer_scores.to_numpy(dtype=float)
        oof_fold[outer_test_index] = outer_fold
        selected_model[outer_test_index] = winner_model
        selected_feature_set[outer_test_index] = winner_set
        selections.append(
            {
                "outer_fold": outer_fold,
                "n_outer_train": len(outer_train_index),
                "n_outer_test": len(outer_test_index),
                "n_passing_screen": len(ranked_features),
                "feature_set": winner_set,
                "n_subtype_features": len(winner_features),
                "model": winner_model,
                "inner_grouped_oof_auc": float(winner["auc"]),
                "selected_features": ";".join(winner_features),
            }
        )

    failure_frame = pd.DataFrame(
        failures, columns=["outer_fold", "feature_set", "model", "error"]
    )
    if not np.isfinite(oof_score).all() or np.any(oof_fold == 0):
        raise ValueError(
            "nested adjusted OOF did not score every development sample; "
            f"failures={failure_frame.to_dict('records')}"
        )

    metrics = pd.DataFrame(
        [
            {
                "split": "train_nested_grouped_oof",
                "dataset": "GSE178341",
                "selection_protocol": "outer_fold_screen_then_inner_grouped_oof",
                "n": len(y),
                "MSI": int(np.sum(y == 1)),
                "MSS": int(np.sum(y == 0)),
                "auc": float(roc_auc_score(y, oof_score)),
            }
        ]
    )
    predictions = pd.DataFrame(
        {
            "split": "train_nested_grouped_oof",
            "dataset": "GSE178341",
            "sampleID": train["sampleID"].astype(str).to_numpy(),
            "patientID": aligned_metadata["patientID"].astype(str).to_numpy(),
            "MMR": train["MMR"].astype(str).to_numpy(),
            "y": y,
            "fold": oof_fold,
            "selected_feature_set": selected_feature_set,
            "selected_model": selected_model,
            "score": oof_score,
        }
    )
    return metrics, predictions, pd.DataFrame(selections), failure_frame


def rank_adjusted_grid(metrics: pd.DataFrame, model_order: list[str]) -> pd.DataFrame:
    required = {"split", "feature_set", "n_subtype_features", "model", "auc"}
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise ValueError(f"adjusted grid metrics are missing columns: {missing}")
    ranked = metrics.loc[metrics["split"].eq("train_grouped_oof")].copy()
    if ranked.empty:
        raise ValueError("adjusted grid has no grouped OOF metrics")
    model_lookup = {model: index for index, model in enumerate(model_order)}
    ranked["model_order"] = ranked["model"].map(model_lookup).fillna(len(model_lookup)).astype(int)
    ranked = ranked.sort_values(
        ["auc", "n_subtype_features", "model_order", "feature_set"],
        ascending=[False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked.insert(0, "train_rank", np.arange(1, len(ranked) + 1))
    return ranked


def _validate_production_contract(
    train: pd.DataFrame,
    external: pd.DataFrame,
    metadata_train: pd.DataFrame,
    metadata_external: pd.DataFrame,
    candidate_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    aligned_train = _align_metadata(train, metadata_train, "development")
    aligned_external = _align_metadata(external, metadata_external, "external")
    contracts = (
        ("GSE178341", train, aligned_train, 64, 62, 35, 29),
        ("GSE132465", external, aligned_external, 23, 23, 4, 19),
    )
    for dataset, features, metadata, n, patients, msi, mss in contracts:
        observed = (
            len(features),
            metadata["patientID"].nunique(),
            int(features["MMR"].eq("MSI").sum()),
            int(features["MMR"].eq("MSS").sum()),
        )
        expected = (n, patients, msi, mss)
        if observed != expected:
            raise ValueError(f"{dataset} cohort contract differs: observed={observed}, expected={expected}")
        if not np.array_equal(
            features["MMR"].astype(str).to_numpy(), metadata["MMR"].astype(str).to_numpy()
        ):
            raise ValueError(f"{dataset} molecular and metadata labels differ")
        _numeric_molecular(features, candidate_features + [MLH1_FEATURE], dataset)
        missing_covariates = sorted(set(ADJUSTMENT_COVARIATES).difference(metadata.columns))
        if missing_covariates:
            raise ValueError(f"{dataset} adjustment covariates are missing: {missing_covariates}")
    if len(candidate_features) != 71 or len(set(candidate_features)) != 71:
        raise ValueError("the unmerged subtype contract must contain 71 unique candidates")
    return aligned_train, aligned_external


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
    return pd.DataFrame(rows)


def _effective_sample_size(weights: pd.Series | np.ndarray) -> float:
    values = np.asarray(weights, dtype=float)
    denominator = float(np.square(values).sum())
    return float(values.sum() ** 2 / denominator) if denominator > 0 else np.nan


def _stratified_bootstrap_weighted_auc(
    y: np.ndarray,
    score: np.ndarray,
    weights: np.ndarray,
    groups: np.ndarray | None,
    n_bootstrap: int,
    random_state: int,
) -> tuple[float, float, float, int]:
    labels = np.asarray(y, dtype=int)
    scores = np.asarray(score, dtype=float)
    sample_weights = np.asarray(weights, dtype=float)
    rng = np.random.default_rng(int(random_state))
    values: list[float] = []
    if groups is None:
        units_by_class = {label: np.flatnonzero(labels == label) for label in (0, 1)}
        for _ in range(int(n_bootstrap)):
            indices = np.concatenate(
                [rng.choice(index, size=len(index), replace=True) for index in units_by_class.values()]
            )
            values.append(weighted_auc(labels[indices], scores[indices], sample_weights[indices]))
    else:
        group_values = np.asarray(groups).astype(str)
        group_frame = pd.DataFrame({"group": group_values, "y": labels})
        if group_frame.groupby("group")["y"].nunique().gt(1).any():
            raise ValueError("a weighted bootstrap group cannot contain both classes")
        group_labels = group_frame.drop_duplicates("group").set_index("group")["y"]
        units_by_class = {
            label: group_labels.index[group_labels.eq(label)].to_numpy(dtype=str)
            for label in (0, 1)
        }
        index_by_group = {
            group: np.flatnonzero(group_values == group) for group in group_labels.index
        }
        for _ in range(int(n_bootstrap)):
            sampled_groups = np.concatenate(
                [
                    rng.choice(group_names, size=len(group_names), replace=True)
                    for group_names in units_by_class.values()
                ]
            )
            indices = np.concatenate([index_by_group[group] for group in sampled_groups])
            values.append(weighted_auc(labels[indices], scores[indices], sample_weights[indices]))
    lower, upper = np.quantile(values, [0.025, 0.975])
    return (
        weighted_auc(labels, scores, sample_weights),
        float(lower),
        float(upper),
        len(values),
    )


def _auc_summary(
    analysis: str,
    dataset: str,
    scores: pd.DataFrame,
    groups: pd.Series | np.ndarray | None = None,
    weights: pd.Series | np.ndarray | None = None,
    estimand: str = "sample_level_auc",
    notes: str = "",
    random_state: int = RANDOM_STATE,
) -> dict[str, object]:
    y = scores["y"].to_numpy(dtype=int)
    values = scores["score"].to_numpy(dtype=float)
    group_values = None if groups is None else np.asarray(groups).astype(str)
    if weights is None:
        auc, lower, upper, success = stratified_bootstrap_auc(
            y,
            values,
            groups=group_values,
            n_bootstrap=N_BOOTSTRAP,
            random_state=random_state,
        )
        effective_n = float(len(scores))
    else:
        weight_values = np.asarray(weights, dtype=float)
        auc, lower, upper, success = _stratified_bootstrap_weighted_auc(
            y,
            values,
            weight_values,
            group_values,
            N_BOOTSTRAP,
            random_state,
        )
        effective_n = _effective_sample_size(weight_values)
    return {
        "analysis": analysis,
        "dataset": dataset,
        "estimand": estimand,
        "n": len(scores),
        "MSI": int(np.sum(y == 1)),
        "MSS": int(np.sum(y == 0)),
        "auc": auc,
        "auc_ci_lower": lower,
        "auc_ci_upper": upper,
        "bootstrap_success": success,
        "effective_sample_size": effective_n,
        "notes": notes,
    }


def _metadata_association_table(metadata_frames: list[pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metadata in metadata_frames:
        dataset = str(metadata["dataset"].iloc[0])
        y = metadata["y"].to_numpy(dtype=int)
        for covariate in ADJUSTMENT_COVARIATES:
            values = pd.to_numeric(metadata[covariate], errors="coerce")
            valid = values.notna()
            if covariate in CONTINUOUS_COVARIATES:
                msi = values[valid & (y == 1)]
                mss = values[valid & (y == 0)]
                p_value = float(mannwhitneyu(msi, mss, alternative="two-sided").pvalue)
                effect = float(msi.mean() - mss.mean())
                msi_summary = float(msi.mean())
                mss_summary = float(mss.mean())
                test = "Mann-Whitney U"
                variable_type = "continuous"
            else:
                table = pd.crosstab(y[valid], values[valid]).reindex(
                    index=[0, 1], columns=[0.0, 1.0], fill_value=0
                )
                _, p_value = fisher_exact(table.to_numpy())
                msi_summary = float(values[valid & (y == 1)].mean())
                mss_summary = float(values[valid & (y == 0)].mean())
                effect = msi_summary - mss_summary
                test = "Fisher exact"
                variable_type = "binary"
            rows.append(
                {
                    "dataset": dataset,
                    "covariate": covariate,
                    "variable_type": variable_type,
                    "n_nonmissing": int(valid.sum()),
                    "n_missing": int((~valid).sum()),
                    "MSI_mean_or_proportion": msi_summary,
                    "MSS_mean_or_proportion": mss_summary,
                    "MSI_minus_MSS": effect,
                    "test": test,
                    "p_value": p_value,
                }
            )

        if dataset == "GSE178341":
            for covariate in ("source_hospital", "processing_team", "chemistry"):
                table = pd.crosstab(metadata["MMR"], metadata[covariate].fillna("missing"))
                if table.shape[0] == 2 and table.shape[1] > 1:
                    statistic, p_value, _, _ = chi2_contingency(table)
                    denominator = len(metadata) * min(table.shape[0] - 1, table.shape[1] - 1)
                    cramer_v = float(np.sqrt(statistic / denominator)) if denominator else np.nan
                    reason = ""
                else:
                    p_value = np.nan
                    cramer_v = np.nan
                    reason = "fewer than two outcome or metadata levels"
                rows.append(
                    {
                        "dataset": dataset,
                        "covariate": covariate,
                        "variable_type": "categorical_internal_audit_only",
                        "n_nonmissing": int(metadata[covariate].notna().sum()),
                        "n_missing": int(metadata[covariate].isna().sum()),
                        "MSI_mean_or_proportion": np.nan,
                        "MSS_mean_or_proportion": np.nan,
                        "MSI_minus_MSS": cramer_v,
                        "test": "chi-square; effect=Cramer's V",
                        "p_value": p_value,
                        "reason": reason,
                    }
                )
    return pd.DataFrame(rows)


def _covariate_balance_table(
    metadata: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.DataFrame:
    if len(metadata) != len(weights):
        raise ValueError("metadata and overlap weights have different row counts")
    weight_values = weights["overlap_weight"].to_numpy(dtype=float)
    propensity = weights["propensity"].to_numpy(dtype=float)
    y = metadata["y"].to_numpy(dtype=int)
    rows = []
    for covariate in ADJUSTMENT_COVARIATES:
        values = pd.to_numeric(metadata[covariate], errors="coerce").to_numpy(dtype=float)
        rows.append(
            {
                "dataset": metadata["dataset"].iloc[0],
                "covariate": covariate,
                "unweighted_smd": standardized_mean_difference(values, y),
                "overlap_weighted_smd": standardized_mean_difference(values, y, weight_values),
                "n_missing": int(np.sum(~np.isfinite(values))),
                "effective_sample_size": _effective_sample_size(weight_values),
                "propensity_min": float(np.min(propensity)),
                "propensity_max": float(np.max(propensity)),
            }
        )
    return pd.DataFrame(rows)


def _attach_patient_ids(scores: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    patient_by_sample = metadata.set_index(metadata["sampleID"].astype(str))["patientID"]
    result = scores.copy()
    result["patientID"] = result["sampleID"].astype(str).map(patient_by_sample)
    if result["patientID"].isna().any():
        missing = result.loc[result["patientID"].isna(), "sampleID"].astype(str).tolist()
        raise ValueError(f"scores are missing patient mappings: {missing}")
    return result


def build_current_model_audit(
    train: pd.DataFrame,
    external: pd.DataFrame,
    metadata_train: pd.DataFrame,
    metadata_external: pd.DataFrame,
    current_predictions: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reproduce and audit the locked expression-only shared18+MLH1 model."""

    model_features = list(CURRENT_SHARED18) + [MLH1_FEATURE]
    _, grouped_scores = evaluate_unadjusted_candidate(
        train,
        external,
        metadata_train,
        metadata_external,
        model_features,
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        "lda_shrinkage",
        folds,
        RANDOM_STATE,
    )
    archived = current_predictions.loc[
        current_predictions["feature_set"].eq("top_18_plus_mlh1")
        & current_predictions["model"].eq("lda_shrinkage")
        & current_predictions["split"].isin(["train_5fold_oof", "external"]),
        ["split", "dataset", "sampleID", "MMR", "y", "fold", "score"],
    ].copy()
    if archived.groupby("split").size().to_dict() != {"external": 23, "train_5fold_oof": 64}:
        raise ValueError("archived shared18+MLH1 prediction rows do not match the cohort contract")
    archived_train = _attach_patient_ids(
        archived.loc[archived["split"].eq("train_5fold_oof")].reset_index(drop=True),
        metadata_train,
    )
    archived_external = _attach_patient_ids(
        archived.loc[archived["split"].eq("external")].reset_index(drop=True),
        metadata_external,
    )
    archived_auc = roc_auc_score(archived_train["y"], archived_train["score"])
    if not np.isclose(archived_auc, 0.9665024630541872, atol=1e-12):
        raise ValueError(f"current archived OOF AUROC changed: {archived_auc}")

    grouped_train = grouped_scores.loc[
        grouped_scores["split"].eq("train_grouped_oof")
    ].reset_index(drop=True)
    grouped_external = grouped_scores.loc[
        grouped_scores["split"].eq("external")
    ].reset_index(drop=True)
    external_difference = np.max(
        np.abs(
            grouped_external.set_index("sampleID").loc[
                archived_external["sampleID"], "score"
            ].to_numpy()
            - archived_external["score"].to_numpy()
        )
    )
    if external_difference > 1e-12:
        raise ValueError(f"current external scores changed by {external_difference}")

    train_weights = overlap_weights(
        metadata_train, ADJUSTMENT_COVARIATES, patient_weight=True, random_state=RANDOM_STATE
    )
    external_weights = overlap_weights(
        metadata_external, ADJUSTMENT_COVARIATES, patient_weight=True, random_state=RANDOM_STATE
    )
    train_weight_by_sample = pd.Series(
        train_weights["overlap_weight"].to_numpy(),
        index=metadata_train["sampleID"].astype(str),
    )
    external_weight_by_sample = pd.Series(
        external_weights["overlap_weight"].to_numpy(),
        index=metadata_external["sampleID"].astype(str),
    )
    rows = [
        _auc_summary(
            "current_archived_fixed_screen_oof",
            "GSE178341",
            archived_train,
            groups=archived_train["patientID"],
            notes="Original development-wide feature screen and ordinary five-fold OOF",
            random_state=2026,
        ),
        _auc_summary(
            "current_patient_grouped_oof",
            "GSE178341",
            grouped_train,
            groups=grouped_train["patientID"],
            notes="Fixed shared18+MLH1; patients kept within folds",
            random_state=2026,
        ),
        _auc_summary(
            "current_external",
            "GSE132465",
            grouped_external,
            notes="Full GSE178341 fit evaluated in GSE132465",
            random_state=2026,
        ),
    ]

    patient_average = (
        grouped_train.groupby("patientID", as_index=False)
        .agg(y=("y", "first"), score=("score", "mean"))
    )
    rows.append(
        _auc_summary(
            "current_patient_grouped_oof_patient_average",
            "GSE178341",
            patient_average,
            estimand="patient_level_auc",
            notes="Multi-region scores averaged within patient",
            random_state=2026,
        )
    )
    for analysis, dataset, score_frame, metadata, seed in (
        (
            "current_right_sided_grouped_oof",
            "GSE178341",
            grouped_train,
            metadata_train,
            2026,
        ),
        (
            "current_right_sided_external",
            "GSE132465",
            grouped_external,
            metadata_external,
            2026,
        ),
    ):
        right_samples = set(
            metadata.loc[metadata["right_sided"].eq(1), "sampleID"].astype(str)
        )
        subset = score_frame.loc[score_frame["sampleID"].astype(str).isin(right_samples)].copy()
        group_values = subset["patientID"] if dataset == "GSE178341" else None
        rows.append(
            _auc_summary(
                analysis,
                dataset,
                subset,
                groups=group_values,
                notes=(
                    "All external MSI cases are right-sided; this is a restricted sensitivity summary"
                    if dataset == "GSE132465"
                    else "Restricted to right-sided tumors"
                ),
                random_state=seed,
            )
        )
    rows.extend(
        [
            _auc_summary(
                "current_archived_oof_overlap_weighted",
                "GSE178341",
                archived_train,
                groups=archived_train["patientID"],
                weights=archived_train["sampleID"].map(train_weight_by_sample),
                estimand="overlap_weighted_sample_level_auc",
                notes="Overlap weighted on all six harmonized clinical covariates",
                random_state=2026,
            ),
            _auc_summary(
                "current_patient_grouped_oof_overlap_weighted",
                "GSE178341",
                grouped_train,
                groups=grouped_train["patientID"],
                weights=grouped_train["sampleID"].map(train_weight_by_sample),
                estimand="overlap_weighted_sample_level_auc",
                notes="Patient-grouped OOF scores; overlap weighted on all six clinical covariates",
                random_state=2026,
            ),
            _auc_summary(
                "current_external_overlap_weighted",
                "GSE132465",
                grouped_external,
                weights=grouped_external["sampleID"].map(external_weight_by_sample),
                estimand="overlap_weighted_sample_level_auc",
                notes="Limited overlap because all four MSI tumors are right-sided",
                random_state=2026,
            ),
        ]
    )

    score_output = pd.concat(
        [
            archived_train.assign(analysis="current_archived_fixed_screen_oof"),
            grouped_train.assign(analysis="current_patient_grouped_oof"),
            grouped_external.assign(analysis="current_external"),
        ],
        ignore_index=True,
    )[
        ["analysis", "split", "dataset", "sampleID", "patientID", "MMR", "y", "fold", "score"]
    ]
    train_weights_output = train_weights.copy()
    train_weights_output.insert(0, "sampleID", metadata_train["sampleID"].astype(str).to_numpy())
    train_weights_output.insert(0, "dataset", "GSE178341")
    external_weights_output = external_weights.copy()
    external_weights_output.insert(0, "sampleID", metadata_external["sampleID"].astype(str).to_numpy())
    external_weights_output.insert(0, "dataset", "GSE132465")
    balance = pd.concat(
        [
            _covariate_balance_table(metadata_train, train_weights),
            _covariate_balance_table(metadata_external, external_weights),
        ],
        ignore_index=True,
    )
    return (
        pd.DataFrame(rows),
        score_output,
        balance,
        train_weights_output,
        external_weights_output,
    )


def _clinical_baseline_with_intervals(
    train: pd.DataFrame,
    external: pd.DataFrame,
    metadata_train: pd.DataFrame,
    metadata_external: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, scores = evaluate_clinical_baseline(
        train,
        external,
        metadata_train,
        metadata_external,
        folds,
        list(CLINICAL_BASELINE_COVARIATES),
        RANDOM_STATE,
    )
    train_scores = scores.loc[scores["split"].eq("train_grouped_oof")].reset_index(drop=True)
    external_scores = scores.loc[scores["split"].eq("external")].reset_index(drop=True)
    metrics = pd.DataFrame(
        [
            _auc_summary(
                "clinical_only_patient_grouped_oof",
                "GSE178341",
                train_scores,
                groups=train_scores["patientID"],
                notes="age, sex, side, pT group, and node status",
                random_state=2026,
            ),
            _auc_summary(
                "clinical_only_external",
                "GSE132465",
                external_scores,
                notes="age, sex, side, pT group, and node status",
                random_state=2026,
            ),
        ]
    )
    return metrics, scores


def build_locked_shared18_adjustment_audit(
    train: pd.DataFrame,
    external: pd.DataFrame,
    metadata_train: pd.DataFrame,
    metadata_external: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold the shared18 panel and LDA fixed while varying adjustment covariates."""

    metric_rows: list[dict[str, object]] = []
    score_parts: list[pd.DataFrame] = []
    model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    for set_index, (adjustment_set, (continuous, binary)) in enumerate(
        ADJUSTMENT_SET_SPECS.items(), start=1
    ):
        _, scores, _ = evaluate_adjusted_candidate(
            train,
            external,
            metadata_train,
            metadata_external,
            list(CURRENT_SHARED18),
            model,
            "shared18_plus_mlh1",
            "lda_shrinkage",
            folds,
            RANDOM_STATE,
            NUISANCE_ALPHA,
            tuple(continuous),
            tuple(binary),
        )
        covariates = list(continuous) + list(binary)
        for split_index, split in enumerate(("train_grouped_oof", "external"), start=1):
            subset = scores.loc[scores["split"].eq(split)].reset_index(drop=True)
            dataset = str(subset["dataset"].iloc[0])
            groups = subset["patientID"] if dataset == "GSE178341" else None
            summary = _auc_summary(
                f"locked_shared18_{adjustment_set}_{split}",
                dataset,
                subset,
                groups=groups,
                notes="Same shared18+MLH1 panel and shrinkage LDA; only nuisance covariates vary",
                random_state=2026,
            )
            summary.update(
                {
                    "adjustment_set": adjustment_set,
                    "continuous_covariates": ";".join(continuous),
                    "binary_covariates": ";".join(binary),
                    "n_adjustment_covariates": len(covariates),
                    "model": "lda_shrinkage",
                    "n_model_features": 19,
                }
            )
            metric_rows.append(summary)
        score_parts.append(
            scores.loc[scores["split"].isin(["train_grouped_oof", "external"])].assign(
                adjustment_set=adjustment_set,
                adjustment_covariates=";".join(covariates),
            )
        )
    score_columns = [
        "adjustment_set",
        "adjustment_covariates",
        "split",
        "dataset",
        "feature_set",
        "model",
        "sampleID",
        "patientID",
        "MMR",
        "y",
        "fold",
        "score",
    ]
    return pd.DataFrame(metric_rows), pd.concat(score_parts, ignore_index=True)[score_columns]


def residualize_external_for_direction_audit(
    train: pd.DataFrame,
    external: pd.DataFrame,
    metadata_train: pd.DataFrame,
    metadata_external: pd.DataFrame,
    candidate_features: list[str],
    alpha: float = NUISANCE_ALPHA,
) -> pd.DataFrame:
    """Residualize every candidate for the external univariate direction audit."""

    molecular_features = list(candidate_features) + [MLH1_FEATURE]
    aligned_train = _align_metadata(train, metadata_train, "development")
    aligned_external = _align_metadata(external, metadata_external, "external")
    x_train = _numeric_molecular(train, molecular_features, "development")
    x_external = _numeric_molecular(external, molecular_features, "external")
    residualizer = MetadataResidualizer(alpha=alpha)
    residualizer.fit(aligned_train, x_train)
    return residualizer.transform(aligned_external, x_external)


def run_complete_adjusted_analysis(
    train: pd.DataFrame,
    external: pd.DataFrame,
    metadata_train: pd.DataFrame,
    metadata_external: pd.DataFrame,
    candidate_features: list[str],
    class_to_group: dict[str, str],
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, pd.DataFrame], dict[str, object], dict[str, object]]:
    """Run fixed-screen and nested metadata-adjusted sensitivity analyses."""

    all_molecular_features = candidate_features + [MLH1_FEATURE]
    crossfit_residuals, fold_assignment = cross_fitted_residuals(
        train,
        metadata_train,
        all_molecular_features,
        folds,
        NUISANCE_ALPHA,
    )
    screening_frame = crossfit_residuals.loc[:, candidate_features].copy()
    screening_frame.insert(0, "MMR", train["MMR"].astype(str).to_numpy())
    adjusted_stats = screen_features(screening_frame, candidate_features, class_to_group)
    passing_stats = adjusted_stats.loc[
        adjusted_stats["auc_best_direction"].gt(FEATURE_THRESHOLD)
    ].reset_index(drop=True)
    ranked_features = passing_stats["feature"].astype(str).tolist()
    feature_sets = build_adjusted_feature_sets(
        ranked_features,
        MINIMUM_FEATURE_COUNT,
        MAXIMUM_FEATURE_COUNT,
    )
    models = make_models(RANDOM_STATE)
    grid_metrics, grid_predictions, _, grid_failures = run_adjusted_model_grid(
        train,
        external,
        metadata_train,
        metadata_external,
        feature_sets,
        models,
        folds,
        RANDOM_STATE,
        NUISANCE_ALPHA,
    )
    if not grid_failures.empty:
        raise ValueError(f"adjusted fixed-screen model grid failures: {grid_failures.to_dict('records')}")
    ranked_grid = rank_adjusted_grid(grid_metrics, list(models))
    winner = ranked_grid.iloc[0]
    winner_set = str(winner["feature_set"])
    winner_model = str(winner["model"])
    winner_features = feature_sets[winner_set]
    selected_metrics_raw = grid_metrics.loc[
        grid_metrics["feature_set"].eq(winner_set)
        & grid_metrics["model"].eq(winner_model)
    ].copy()
    selected_scores = grid_predictions.loc[
        grid_predictions["feature_set"].eq(winner_set)
        & grid_predictions["model"].eq(winner_model)
    ].copy()

    nested_metrics_raw, nested_scores, nested_selections, nested_failures = (
        run_nested_adjusted_oof(
            train,
            metadata_train,
            candidate_features,
            models,
            folds,
            FEATURE_THRESHOLD,
            MINIMUM_FEATURE_COUNT,
            MAXIMUM_FEATURE_COUNT,
            N_SPLITS,
            RANDOM_STATE,
            NUISANCE_ALPHA,
        )
    )
    if not nested_failures.empty:
        raise ValueError(f"adjusted nested model failures: {nested_failures.to_dict('records')}")

    selected_summary_rows = []
    for offset, split in enumerate(("train_grouped_oof", "train_full_fit", "external"), start=1):
        subset = selected_scores.loc[selected_scores["split"].eq(split)].reset_index(drop=True)
        groups = subset["patientID"] if subset["dataset"].iloc[0] == "GSE178341" else None
        notes = (
            "Development-wide adjusted feature screen; patient-grouped OOF"
            if split == "train_grouped_oof"
            else (
                "Resubstitution score after full-development fit"
                if split == "train_full_fit"
                else "Model selected without using external AUROC"
            )
        )
        row = _auc_summary(
            f"adjusted_selected_fixed_screen_{split}",
            str(subset["dataset"].iloc[0]),
            subset,
            groups=groups,
            notes=notes,
            random_state=2026,
        )
        row.update(
            {
                "feature_set": winner_set,
                "n_subtype_features": len(winner_features),
                "n_model_features": len(winner_features) + 1,
                "model": winner_model,
            }
        )
        selected_summary_rows.append(row)
    nested_row = _auc_summary(
        "adjusted_fully_nested_patient_grouped_oof",
        "GSE178341",
        nested_scores,
        groups=nested_scores["patientID"],
        notes="Feature screening, feature count, and classifier selected inside each outer fold",
        random_state=2026,
    )
    nested_row.update(
        {
            "feature_set": "outer_fold_specific",
            "n_subtype_features": np.nan,
            "n_model_features": np.nan,
            "model": "outer_fold_specific",
        }
    )
    selected_summary_rows.append(nested_row)
    selected_summary = pd.DataFrame(selected_summary_rows)
    nested_summary = pd.DataFrame([nested_row])

    full_residualizer = MetadataResidualizer(alpha=NUISANCE_ALPHA)
    selected_model_features = winner_features + [MLH1_FEATURE]
    x_train = _numeric_molecular(train, selected_model_features, "development")
    x_external = _numeric_molecular(external, selected_model_features, "external")
    full_residualizer.fit(metadata_train, x_train)
    residual_train = full_residualizer.transform(metadata_train, x_train)
    residual_external = full_residualizer.transform(metadata_external, x_external)
    full_scaler = MinMaxScaler().fit(residual_train.to_numpy(dtype=float))
    full_classifier = clone(models[winner_model])
    full_classifier.fit(
        full_scaler.transform(residual_train.to_numpy(dtype=float)),
        train["MMR"].astype(str).eq("MSI").to_numpy(dtype=int),
    )
    final_external_score, _, _ = _score_and_predict(
        full_classifier, full_scaler.transform(residual_external.to_numpy(dtype=float))
    )
    selected_external = selected_scores.loc[selected_scores["split"].eq("external")]
    if not np.allclose(final_external_score, selected_external["score"].to_numpy(), atol=1e-12):
        raise ValueError("serialized sensitivity model does not reproduce selected external scores")

    external_direction_residuals = residualize_external_for_direction_audit(
        train,
        external,
        metadata_train,
        metadata_external,
        candidate_features,
        NUISANCE_ALPHA,
    )
    external_residual_frame = external_direction_residuals.loc[:, candidate_features].copy()
    external_residual_frame.insert(0, "MMR", external["MMR"].astype(str).to_numpy())
    external_adjusted_stats = screen_features(
        external_residual_frame, candidate_features, class_to_group
    )
    unadjusted_stats = screen_features(train, candidate_features, class_to_group)
    comparison = (
        pd.DataFrame({"feature": list(CURRENT_SHARED18), "shared18_rank": range(1, 19)})
        .merge(
            unadjusted_stats[
                ["feature", "auc_best_direction", "effect_direction", "direction_sign"]
            ].rename(
                columns={
                    "auc_best_direction": "unadjusted_train_auc_best_direction",
                    "effect_direction": "unadjusted_train_effect_direction",
                    "direction_sign": "unadjusted_train_direction_sign",
                }
            ),
            on="feature",
            how="left",
        )
        .merge(
            adjusted_stats[
                ["feature", "auc_best_direction", "effect_direction", "direction_sign"]
            ].rename(
                columns={
                    "auc_best_direction": "adjusted_train_auc_best_direction",
                    "effect_direction": "adjusted_train_effect_direction",
                    "direction_sign": "adjusted_train_direction_sign",
                }
            ),
            on="feature",
            how="left",
        )
        .merge(
            external_adjusted_stats[
                ["feature", "auc_best_direction", "effect_direction", "direction_sign"]
            ].rename(
                columns={
                    "auc_best_direction": "adjusted_external_auc_best_direction",
                    "effect_direction": "adjusted_external_effect_direction",
                    "direction_sign": "adjusted_external_direction_sign",
                }
            ),
            on="feature",
            how="left",
        )
    )
    comparison["passes_adjusted_train_auc_gt065"] = comparison["feature"].isin(ranked_features)
    comparison["selected_sensitivity_model"] = comparison["feature"].isin(winner_features)
    comparison["unadjusted_vs_adjusted_train_same_direction"] = (
        comparison["unadjusted_train_direction_sign"]
        == comparison["adjusted_train_direction_sign"]
    )
    comparison["adjusted_train_vs_external_same_direction"] = (
        comparison["adjusted_train_direction_sign"]
        == comparison["adjusted_external_direction_sign"]
    )

    fixed_metric_columns = [
        "split",
        "dataset",
        "feature_set",
        "n_subtype_features",
        "n_model_features",
        "model",
        "normalization",
        "n",
        "MSI",
        "MSS",
        "auc",
    ]
    score_columns = [
        "split",
        "dataset",
        "feature_set",
        "n_subtype_features",
        "n_model_features",
        "model",
        "sampleID",
        "patientID",
        "MMR",
        "y",
        "fold",
        "score",
    ]
    selected_score_output = pd.concat(
        [
            selected_scores.loc[:, score_columns].assign(selection="fixed_screen_selected"),
            nested_scores.assign(
                feature_set="outer_fold_specific",
                n_subtype_features=np.nan,
                n_model_features=np.nan,
                model="outer_fold_specific",
                selection="fully_nested",
            ).loc[:, score_columns + ["selection"]],
        ],
        ignore_index=True,
    )
    tables = {
        "adjusted_univariate_AUROC_71.csv": adjusted_stats,
        "adjusted_AUROC_gt065_features.csv": passing_stats,
        "adjusted_feature_sets.csv": _feature_set_table(feature_sets),
        "adjusted_fixed_screen_model_grid_AUROC.csv": grid_metrics.loc[:, fixed_metric_columns],
        "adjusted_fixed_screen_sample_scores.csv": grid_predictions.loc[:, score_columns],
        "adjusted_nested_OOF_AUROC.csv": nested_summary,
        "adjusted_nested_fold_selections.csv": nested_selections,
        "adjusted_selected_model_AUROC.csv": selected_summary,
        "adjusted_selected_model_sample_scores.csv": selected_score_output,
        "shared18_adjusted_overlap_and_direction.csv": comparison,
    }
    selected = {
        "feature_set": winner_set,
        "n_subtype_features": len(winner_features),
        "n_model_features": len(winner_features) + 1,
        "subtype_features": winner_features,
        "model": winner_model,
        "train_grouped_oof_auc": float(
            selected_metrics_raw.loc[
                selected_metrics_raw["split"].eq("train_grouped_oof"), "auc"
            ].iloc[0]
        ),
        "external_auc": float(
            selected_metrics_raw.loc[selected_metrics_raw["split"].eq("external"), "auc"].iloc[0]
        ),
        "nested_oof_auc": float(nested_metrics_raw["auc"].iloc[0]),
        "n_adjusted_auc_gt065": len(ranked_features),
    }
    artifact = {
        "analysis_role": "metadata-adjusted sensitivity model",
        "primary_model_replaced": False,
        "covariates": list(ADJUSTMENT_COVARIATES),
        "nuisance_alpha": NUISANCE_ALPHA,
        "subtype_features": winner_features,
        "model_features": selected_model_features,
        "model_name": winner_model,
        "residualizer": full_residualizer,
        "scaler": full_scaler,
        "classifier": full_classifier,
    }
    diagnostics = {
        "fixed_grid_failures": int(len(grid_failures)),
        "nested_failures": int(len(nested_failures)),
        "crossfit_fold_assignment": fold_assignment,
    }
    return tables, selected, {"model_artifact": artifact, **diagnostics}


def _decision_and_summary(
    current_metrics: pd.DataFrame,
    adjusted_metrics: pd.DataFrame,
    locked_adjustment_metrics: pd.DataFrame,
    clinical_metrics: pd.DataFrame,
    association: pd.DataFrame,
    selected: dict[str, object],
    overlap: pd.DataFrame,
) -> tuple[str, str]:
    def row(frame: pd.DataFrame, analysis: str) -> pd.Series:
        selected_rows = frame.loc[frame["analysis"].eq(analysis)]
        if len(selected_rows) != 1:
            raise ValueError(f"expected one summary row for {analysis}")
        return selected_rows.iloc[0]

    current_development = row(current_metrics, "current_patient_grouped_oof")
    current_external = row(current_metrics, "current_external")
    adjusted_development = row(
        adjusted_metrics, "adjusted_fully_nested_patient_grouped_oof"
    )
    adjusted_external = row(
        adjusted_metrics, "adjusted_selected_fixed_screen_external"
    )
    locked_adjusted_development = row(
        locked_adjustment_metrics,
        "locked_shared18_all_clinical_train_grouped_oof",
    )
    locked_adjusted_external = row(
        locked_adjustment_metrics,
        "locked_shared18_all_clinical_external",
    )
    clinical_development = row(clinical_metrics, "clinical_only_patient_grouped_oof")
    clinical_external = row(clinical_metrics, "clinical_only_external")

    development_compatible = not (
        adjusted_development["auc_ci_upper"] < current_development["auc_ci_lower"]
        or current_development["auc_ci_upper"] < adjusted_development["auc_ci_lower"]
    )
    external_compatible = not (
        adjusted_external["auc_ci_upper"] < current_external["auc_ci_lower"]
        or current_external["auc_ci_upper"] < adjusted_external["auc_ci_lower"]
    )
    exceeds_clinical = (
        adjusted_development["auc"] > clinical_development["auc"]
        and adjusted_external["auc"] > clinical_external["auc"]
    )
    if development_compatible and external_compatible and exceeds_clinical:
        decision = "robust sensitivity"
        decision_cn = "稳健性敏感性结果"
        interpretation = (
            "校正后的完全嵌套开发集估计及外部估计仍与当前表达模型相容，"
            "且点估计均高于仅临床变量基线。"
        )
    elif (
        adjusted_development["auc_ci_upper"] < current_development["auc_ci_lower"]
        or adjusted_external["auc_ci_upper"] < current_external["auc_ci_lower"]
    ):
        decision = "material metadata sensitivity"
        decision_cn = "存在实质性元数据敏感性"
        interpretation = "校正后 AUROC 的置信区间与当前模型出现支持实质下降的分离。"
    else:
        decision = "inconclusive"
        decision_cn = "证据尚不确定"
        interpretation = (
            "样本量、外部阳性例数或置信区间不足以区分稳健性与实质性元数据影响。"
        )

    significant = association.loc[
        association["p_value"].lt(0.05)
        & association["variable_type"].isin(["continuous", "binary"]),
        ["dataset", "covariate", "p_value"],
    ]
    imbalance_text = (
        "；".join(
            f"{record.dataset} 的 {record.covariate}（P={record.p_value:.4g}）"
            for record in significant.itertuples(index=False)
        )
        if not significant.empty
        else "未观察到 P<0.05 的单变量元数据差异"
    )
    shared_selected = int(overlap["selected_sensitivity_model"].sum())
    same_direction = int(overlap["unadjusted_vs_adjusted_train_same_direction"].sum())
    summary = f"""# 单细胞模型元数据混杂敏感性分析

## 结论

判定为**{decision_cn}**。{interpretation} 该分析是对现有表达模型的敏感性检验，不替换当前 shared18+MLH1 主模型。

## 元数据审计

开发集包含 64 个肿瘤标本、62 位患者（35 MSI、29 MSS），外部集包含 23 位患者（4 MSI、19 MSS）。观察到的显著单变量差异包括：{imbalance_text}。这些差异说明需要做混杂敏感性分析，但单变量关联本身不能证明模型受其驱动。

## 模型结果

- 当前 shared18+MLH1 模型：患者分组 OOF AUROC {current_development['auc']:.3f}（95% CI {current_development['auc_ci_lower']:.3f}-{current_development['auc_ci_upper']:.3f}）；GSE132465 AUROC {current_external['auc']:.3f}（95% CI {current_external['auc_ci_lower']:.3f}-{current_external['auc_ci_upper']:.3f}）。
- 固定相同 shared18+MLH1 与 shrinkage LDA、仅加入全部六项临床协变量残差化后，患者分组 OOF AUROC 为 {locked_adjusted_development['auc']:.3f}（95% CI {locked_adjusted_development['auc_ci_lower']:.3f}-{locked_adjusted_development['auc_ci_upper']:.3f}），外部 AUROC 为 {locked_adjusted_external['auc']:.3f}（95% CI {locked_adjusted_external['auc_ci_lower']:.3f}-{locked_adjusted_external['auc_ci_upper']:.3f}）。该成对结果用于分离校正效应与重新筛特征/选模型效应。
- 校正后共有 {selected['n_adjusted_auc_gt065']} 个亚型的方向校正单变量 AUROC>0.65。开发集内选择的敏感性模型为 {selected['model']}，包含 {selected['n_subtype_features']} 个亚型特征及恶性上皮 MLH1。
- 校正后完全嵌套患者分组 OOF AUROC {adjusted_development['auc']:.3f}（95% CI {adjusted_development['auc_ci_lower']:.3f}-{adjusted_development['auc_ci_upper']:.3f}）；固定开发集筛选后在 GSE132465 的 AUROC {adjusted_external['auc']:.3f}（95% CI {adjusted_external['auc_ci_lower']:.3f}-{adjusted_external['auc_ci_upper']:.3f}）。
- 仅临床变量基线：开发集患者分组 OOF AUROC {clinical_development['auc']:.3f}，外部 AUROC {clinical_external['auc']:.3f}。
- 校正后最终敏感性模型与原 shared18 面板重叠 {shared_selected}/18 个特征；原始与校正后开发集方向一致 {same_direction}/18 个。

## 解释边界

残差化仅使用年龄、性别、肿瘤侧别、pT 分组、淋巴结状态和病理分级，并在每个训练折内拟合。总肿瘤细胞数和用于 MLH1 估计的上皮细胞数未作为校正变量，因为本研究的分子输入本身包含细胞亚型比例，校正细胞数量可能去除研究对象的一部分。性能变化说明模型对已测量临床变量敏感，但不能单独证明因果混杂或给出“去混杂后的真实效应”。GSE132465 的 4 个 MSI 病例全部为右侧结肠肿瘤，因此该队列无法识别完全消除侧别影响后的外部性能；加权与右侧限定结果只能作为敏感性证据。
"""
    return decision, summary


def main() -> None:
    source_files = {
        "GSE178341_molecular_features": {"path": str(TRAIN_FEATURES_PATH)},
        "GSE132465_molecular_features": {"path": str(EXTERNAL_FEATURES_PATH)},
        "unmerged71_class_contract": {"path": str(CLASS_CONTRACT_PATH)},
        "current_model_predictions": {"path": str(CURRENT_PREDICTIONS_PATH)},
        "GSE178341_official_metadata": {
            "path": str(GSE178341_METADATA_PATH),
            "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE178nnn/GSE178341/suppl/GSE178341_crc10x_full_c295v4_submit_metatables.csv.gz",
        },
        "GSE132465_publication_metadata": {
            "path": str(GSE132465_METADATA_PATH),
            "url": "https://static-content.springer.com/esm/art%3A10.1038%2Fs41588-020-0636-z/MediaObjects/41588_2020_636_MOESM3_ESM.xlsx",
        },
    }
    if MANUSCRIPT_PATH.is_file():
        source_files["current_manuscript_read_only"] = {"path": str(MANUSCRIPT_PATH)}
    hashes_before = {
        name: _sha256(record["path"]) for name, record in source_files.items()
    }
    manuscript_mtime_before = (
        MANUSCRIPT_PATH.stat().st_mtime_ns if MANUSCRIPT_PATH.is_file() else None
    )

    train = pd.read_csv(TRAIN_FEATURES_PATH)
    external = pd.read_csv(EXTERNAL_FEATURES_PATH)
    class_contract = pd.read_csv(CLASS_CONTRACT_PATH)
    candidate_features = class_contract["class_name"].astype(str).tolist()
    class_to_group = dict(
        zip(
            class_contract["class_name"].astype(str),
            class_contract["group_class"].astype(str),
        )
    )
    metadata_train = parse_gse178341_metadata(GSE178341_METADATA_PATH)
    metadata_external = parse_gse132465_metadata(GSE132465_METADATA_PATH)
    metadata_train, metadata_external = _validate_production_contract(
        train,
        external,
        metadata_train,
        metadata_external,
        candidate_features,
    )
    folds = make_patient_grouped_folds(
        metadata_train["y"],
        metadata_train["patientID"],
        N_SPLITS,
        RANDOM_STATE,
    )
    for train_index, test_index in folds:
        if set(metadata_train.iloc[train_index]["patientID"]).intersection(
            metadata_train.iloc[test_index]["patientID"]
        ):
            raise ValueError("a patient crosses development folds")

    current_metrics, current_scores, balance, train_weights, external_weights = (
        build_current_model_audit(
            train,
            external,
            metadata_train,
            metadata_external,
            pd.read_csv(CURRENT_PREDICTIONS_PATH),
            folds,
        )
    )
    clinical_metrics, _ = _clinical_baseline_with_intervals(
        train, external, metadata_train, metadata_external, folds
    )
    locked_adjustment_metrics, locked_adjustment_scores = (
        build_locked_shared18_adjustment_audit(
            train, external, metadata_train, metadata_external, folds
        )
    )
    association = _metadata_association_table([metadata_train, metadata_external])
    adjusted_tables, selected, adjusted_diagnostics = run_complete_adjusted_analysis(
        train,
        external,
        metadata_train,
        metadata_external,
        candidate_features,
        class_to_group,
        folds,
    )

    train_weight_columns = train_weights.loc[
        :, ["sampleID", "propensity", "base_patient_weight", "overlap_weight"]
    ].rename(
        columns={
            "propensity": "overlap_propensity",
            "base_patient_weight": "overlap_base_patient_weight",
        }
    )
    external_weight_columns = external_weights.loc[
        :, ["sampleID", "propensity", "base_patient_weight", "overlap_weight"]
    ].rename(
        columns={
            "propensity": "overlap_propensity",
            "base_patient_weight": "overlap_base_patient_weight",
        }
    )
    metadata_train_output = metadata_train.merge(
        train_weight_columns, on="sampleID", how="left", validate="one_to_one"
    )
    metadata_external_output = metadata_external.merge(
        external_weight_columns, on="sampleID", how="left", validate="one_to_one"
    )
    metadata_train_output["metadata_source"] = str(GSE178341_METADATA_PATH)
    metadata_external_output["metadata_source"] = str(GSE132465_METADATA_PATH)

    tables = {
        "metadata_GSE178341.csv": metadata_train_output,
        "metadata_GSE132465.csv": metadata_external_output,
        "metadata_association_with_MSI.csv": association,
        "covariate_balance_overlap_weighting.csv": balance,
        "current_model_grouped_and_weighted_AUROC.csv": current_metrics,
        "current_model_grouped_sample_scores.csv": current_scores,
        "clinical_only_baseline_AUROC.csv": clinical_metrics,
        "locked_shared18_adjustment_set_AUROC.csv": locked_adjustment_metrics,
        "locked_shared18_adjustment_set_sample_scores.csv": locked_adjustment_scores,
        **adjusted_tables,
    }
    decision, summary = _decision_and_summary(
        current_metrics,
        adjusted_tables["adjusted_selected_model_AUROC.csv"],
        locked_adjustment_metrics,
        clinical_metrics,
        association,
        selected,
        adjusted_tables["shared18_adjusted_overlap_and_direction.csv"],
    )

    hashes_after = {
        name: _sha256(record["path"]) for name, record in source_files.items()
    }
    if hashes_after != hashes_before:
        changed = [name for name in hashes_before if hashes_before[name] != hashes_after[name]]
        raise ValueError(f"read-only source files changed during analysis: {changed}")
    manuscript_mtime_after = (
        MANUSCRIPT_PATH.stat().st_mtime_ns if MANUSCRIPT_PATH.is_file() else None
    )
    if manuscript_mtime_after != manuscript_mtime_before:
        raise ValueError("the current manuscript modification time changed during analysis")

    OUT.mkdir(parents=True, exist_ok=True)
    model_artifact_path = OUT / "adjusted_selected_sensitivity_model.joblib"
    joblib.dump(adjusted_diagnostics.pop("model_artifact"), model_artifact_path)
    manifest = {
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "analysis": "scRNA metadata-confounding audit and adjusted sensitivity rebuild",
        "output_directory": str(OUT),
        "primary_model_replaced": False,
        "decision": decision,
        "random_state": RANDOM_STATE,
        "n_splits": N_SPLITS,
        "n_bootstrap": N_BOOTSTRAP,
        "nuisance_regression": {"model": "multi-output Ridge", "alpha": NUISANCE_ALPHA},
        "feature_screen": {
            "candidate_contract": "71 unmerged TOSICA subtypes with at least 50 reference cells",
            "criterion": "direction-corrected cross-fitted residual AUROC > 0.65",
            "prefix_scan": f"top {MINIMUM_FEATURE_COUNT} through min({MAXIMUM_FEATURE_COUNT}, n passing)",
        },
        "covariates": list(ADJUSTMENT_COVARIATES),
        "locked_shared18_adjustment_sets": {
            name: list(continuous) + list(binary)
            for name, (continuous, binary) in ADJUSTMENT_SET_SPECS.items()
        },
        "clinical_only_covariates": list(CLINICAL_BASELINE_COVARIATES),
        "excluded_metadata": {
            "complete_AJCC_stage": "GSE178341 provides pathological T stage rather than harmonized complete AJCC stage",
            "metastasis_status": "61 of 64 GSE178341 specimens are Mx or not entered",
            "hospital_processing_team_chemistry": "audited internally but lack a cross-cohort coding system",
            "tumor_and_mlh1_epithelial_cell_counts": "excluded because cell composition is the molecular study target and cell-count adjustment could remove part of that signal",
            "identifiers_batch_labels": "forbidden because they are identifying, outcome-derived, or leakage-prone",
        },
        "cohort_contracts": {
            "GSE178341": {"specimens": 64, "patients": 62, "MSI": 35, "MSS": 29},
            "GSE132465": {"specimens": 23, "patients": 23, "MSI": 4, "MSS": 19},
        },
        "model_order": list(make_models(RANDOM_STATE)),
        "selected_sensitivity_model": selected,
        "sensitivity_model_artifact": {
            "path": str(model_artifact_path),
            "sha256": _sha256(model_artifact_path),
        },
        "current_primary_model": {
            "features": list(CURRENT_SHARED18) + [MLH1_FEATURE],
            "model": "lda_shrinkage",
            "archived_oof_auc_expected": 0.9665024630541872,
        },
        "external_positivity_limitation": (
            "All four GSE132465 MSI tumors are right-sided; complete side-independent "
            "external transportability is not identifiable."
        ),
        "source_files": source_files,
        "read_only_hashes_before": hashes_before,
        "read_only_hashes_after": hashes_after,
        "read_only_sources_unchanged": True,
        "manuscript_mtime_ns_before": manuscript_mtime_before,
        "manuscript_mtime_ns_after": manuscript_mtime_after,
        "diagnostics": {
            key: value
            for key, value in adjusted_diagnostics.items()
            if key != "crossfit_fold_assignment"
        },
        "limitations": [
            "The adjusted model requires metadata at inference and is a sensitivity model only.",
            "Feature screening in the fixed-screen grid uses development-wide cross-fitted residuals; the fully nested OOF is the selection-optimism audit.",
            "The external cohort contains only four MSI tumors and no left-sided MSI tumor.",
            "Residualization addresses measured harmonized covariates, not unmeasured confounding.",
        ],
    }
    write_output_bundle(OUT, tables, manifest)
    adjusted_diagnostics["crossfit_fold_assignment"].to_csv(
        OUT / "adjusted_crossfit_fold_assignment.csv", index=False
    )
    (OUT / "analysis_summary.md").write_text(summary, encoding="utf-8")

    print(f"Output: {OUT}")
    print(f"Decision: {decision}")
    print(
        "Selected adjusted model: "
        f"{selected['model']} with {selected['n_subtype_features']} subtype features + MLH1"
    )
    print(
        f"AUROC grouped OOF={selected['train_grouped_oof_auc']:.6f}, "
        f"nested OOF={selected['nested_oof_auc']:.6f}, external={selected['external_auc']:.6f}"
    )


if __name__ == "__main__":
    main()
