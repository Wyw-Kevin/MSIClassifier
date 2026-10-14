#!/usr/bin/env python3
"""Metadata harmonization and adjustment utilities for the scRNA sensitivity analysis."""

from __future__ import annotations

# Packaging-only path resolution; archived originals retain exact source bytes.
from pathlib import Path as _MSIPackagePath
import sys as _MSIPackageSys
for _MSIPackageParent in _MSIPackagePath(__file__).resolve().parents:
    if (_MSIPackageParent / "project_paths.py").is_file():
        _MSIPackageSys.path.insert(0, str(_MSIPackageParent))
        break
from project_paths import project_root

from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


GSE178341_REQUIRED = {
    "SPECIMEN_TYPE",
    "PatientTypeID",
    "PID",
    "MMRStatus",
    "Age",
    "Sex",
    "TissueSiteSimple",
    "TumorStage",
    "NodeStatusSimple",
    "HistologicGradeSimple",
    "SOURCE_HOSPITAL",
    "TISSUE_PROCESSING_TEAM",
    "SINGLECELL_TYPE",
}

GSE132465_REQUIRED = {
    "Patient",
    "Tumor",
    "Gender",
    "Age",
    "TNM stage",
    "Stage",
    "Anatomic region",
    "Left/Right-sided",
    "MSI",
    "Pathological subtype",
}

CONTINUOUS_COVARIATES = (
    "age",
)
BINARY_COVARIATES = (
    "female",
    "right_sided",
    "t_high",
    "node_positive",
    "grade_high",
)
ADJUSTMENT_COVARIATES = CONTINUOUS_COVARIATES + BINARY_COVARIATES
TECHNICAL_CELL_COUNT_FIELDS = (
    "log_tumor_cells",
    "log_mlh1_epithelial_cells",
)
FORBIDDEN_COVARIATE_TOKENS = (
    "sample",
    "patient",
    "batch",
    "mmr",
    "msi",
    "mlh1status",
    "label",
    "target",
)


def _require_columns(frame: pd.DataFrame, required: set[str], source: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")


def _binary_map(series: pd.Series, mapping: dict[str, float]) -> pd.Series:
    normalized = series.astype("string").str.strip()
    return normalized.map(mapping).astype(float)


def _parse_t_high(series: pd.Series) -> pd.Series:
    t_number = pd.to_numeric(
        series.astype("string").str.extract(r"(?i)t([1-4])", expand=False),
        errors="coerce",
    )
    return pd.Series(
        np.where(t_number.notna(), t_number.ge(3).astype(float), np.nan),
        index=series.index,
        dtype=float,
    )


def _parse_node_positive(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.upper()
    result = pd.Series(np.nan, index=series.index, dtype=float)
    result.loc[text.str.contains(r"\bN0\b", regex=True, na=False)] = 0.0
    result.loc[text.str.contains(r"\bN[1-3][A-C]?\b", regex=True, na=False)] = 1.0
    return result


def _external_grade(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.lower()
    result = pd.Series(np.nan, index=series.index, dtype=float)
    result.loc[text.str.contains("poorly differentiated", regex=False, na=False)] = 1.0
    low = text.str.contains("well differentiated", regex=False, na=False) | text.str.contains(
        "moderately differentiated", regex=False, na=False
    )
    result.loc[low] = 0.0
    return result


def parse_gse178341_metadata(path: str | Path) -> pd.DataFrame:
    """Parse the official cell-level GEO metadata into one row per tumor specimen."""

    cells = pd.read_csv(path, low_memory=False)
    _require_columns(cells, GSE178341_REQUIRED, "GSE178341 metadata")
    tumor = cells.loc[cells["SPECIMEN_TYPE"].astype("string").str.upper().eq("T")].copy()
    if tumor.empty:
        raise ValueError("GSE178341 metadata contains no tumor specimens")

    stable_columns = sorted(GSE178341_REQUIRED.difference({"SPECIMEN_TYPE", "SINGLECELL_TYPE"}))
    conflicts = tumor.groupby("PatientTypeID", observed=True)[stable_columns].nunique(dropna=False)
    conflicting = conflicts.columns[conflicts.gt(1).any(axis=0)].tolist()
    if conflicting:
        raise ValueError(f"GSE178341 specimen metadata conflicts: {conflicting}")

    metadata = (
        tumor.groupby("PatientTypeID", sort=True, observed=True)
        .agg(
            patientID=("PID", "first"),
            MMR_source=("MMRStatus", "first"),
            age=("Age", "first"),
            sex_source=("Sex", "first"),
            side_source=("TissueSiteSimple", "first"),
            t_stage_source=("TumorStage", "first"),
            node_source=("NodeStatusSimple", "first"),
            grade_source=("HistologicGradeSimple", "first"),
            source_hospital=("SOURCE_HOSPITAL", "first"),
            processing_team=("TISSUE_PROCESSING_TEAM", "first"),
            chemistry=(
                "SINGLECELL_TYPE",
                lambda values: ";".join(sorted(set(values.dropna().astype(str)))),
            ),
            n_tumor_cells=("PatientTypeID", "size"),
        )
        .reset_index()
        .rename(columns={"PatientTypeID": "sample_source"})
    )
    metadata.insert(0, "dataset", "GSE178341")
    metadata.insert(1, "sampleID", metadata["sample_source"].str.replace("_", "", regex=False))
    metadata["patientID"] = metadata["patientID"].astype(str)
    metadata["MMR"] = metadata["MMR_source"].map({"MMRd": "MSI", "MMRp": "MSS"})
    metadata["y"] = metadata["MMR"].map({"MSS": 0, "MSI": 1}).astype("Int64")
    metadata["age"] = pd.to_numeric(metadata["age"], errors="coerce")
    metadata["female"] = _binary_map(metadata["sex_source"], {"M": 0.0, "F": 1.0})
    metadata["right_sided"] = _binary_map(
        metadata["side_source"].astype("string").str.lower(), {"left": 0.0, "right": 1.0}
    )
    metadata["t_high"] = _parse_t_high(metadata["t_stage_source"])
    metadata["node_positive"] = _binary_map(
        metadata["node_source"], {"Nneg": 0.0, "Npos": 1.0}
    )
    metadata["grade_high"] = _binary_map(
        metadata["grade_source"].astype("string").str.lower(), {"low": 0.0, "high": 1.0}
    )
    return metadata


def parse_gse132465_metadata(path: str | Path) -> pd.DataFrame:
    """Parse Supplementary Table 1 into the shared clinical covariate schema."""

    source = pd.read_excel(path, sheet_name="Table S1", header=3)
    _require_columns(source, GSE132465_REQUIRED, "GSE132465 Supplementary Table 1")
    metadata = source.loc[
        source["Patient"].astype("string").str.startswith("SMC", na=False)
        & source["Tumor"].astype("string").str.endswith("-T", na=False)
    ].copy()
    if metadata.empty:
        raise ValueError("GSE132465 Supplementary Table 1 contains no SMC tumors")
    if metadata["Tumor"].duplicated().any():
        duplicated = sorted(metadata.loc[metadata["Tumor"].duplicated(False), "Tumor"].astype(str))
        raise ValueError(f"duplicate GSE132465 tumors: {duplicated}")

    result = pd.DataFrame(
        {
            "dataset": "GSE132465",
            "sampleID": metadata["Tumor"].astype(str),
            "patientID": metadata["Patient"].astype(str),
            "MMR_source": metadata["MSI"].astype(str),
            "age": pd.to_numeric(metadata["Age"], errors="coerce"),
            "sex_source": metadata["Gender"].astype("string"),
            "side_source": metadata["Left/Right-sided"].astype("string"),
            "t_stage_source": metadata["TNM stage"].astype("string"),
            "node_source": metadata["TNM stage"].astype("string"),
            "grade_source": metadata["Pathological subtype"].astype("string"),
            "overall_stage_source": metadata["Stage"].astype("string"),
            "anatomic_region": metadata["Anatomic region"].astype("string"),
        }
    )
    result["MMR"] = result["MMR_source"].map({"MSI-H": "MSI", "MSS": "MSS"})
    result["y"] = result["MMR"].map({"MSS": 0, "MSI": 1}).astype("Int64")
    result["female"] = _binary_map(result["sex_source"], {"M": 0.0, "F": 1.0})
    result["right_sided"] = _binary_map(
        result["side_source"].str.lower(), {"left": 0.0, "right": 1.0}
    )
    result["t_high"] = _parse_t_high(result["t_stage_source"])
    result["node_positive"] = _parse_node_positive(result["node_source"])
    result["grade_high"] = _external_grade(result["grade_source"])
    result["n_tumor_cells"] = np.nan
    return result.sort_values("sampleID", kind="mergesort").reset_index(drop=True)


def attach_molecular_cell_yield(
    metadata: pd.DataFrame,
    molecular_features: pd.DataFrame,
) -> pd.DataFrame:
    """Attach epithelial-cell yield and add the two prespecified log-count covariates."""

    required_metadata = {"sampleID", "n_tumor_cells"}
    required_molecular = {"sampleID", "n_epithelial_cells_for_MLH1"}
    _require_columns(metadata, required_metadata, "sample metadata")
    _require_columns(molecular_features, required_molecular, "molecular feature table")
    if metadata["sampleID"].duplicated().any() or molecular_features["sampleID"].duplicated().any():
        raise ValueError("sampleID must be unique before attaching cell-yield covariates")

    epithelial = molecular_features.loc[
        :, ["sampleID", "n_epithelial_cells_for_MLH1"]
    ].copy()
    epithelial["n_epithelial_cells_for_MLH1"] = pd.to_numeric(
        epithelial["n_epithelial_cells_for_MLH1"], errors="coerce"
    )
    result = metadata.merge(epithelial, on="sampleID", how="left", validate="one_to_one")
    if result["n_epithelial_cells_for_MLH1"].isna().any():
        missing = result.loc[result["n_epithelial_cells_for_MLH1"].isna(), "sampleID"].tolist()
        raise ValueError(f"missing epithelial cell counts: {missing}")
    result["n_tumor_cells"] = pd.to_numeric(result["n_tumor_cells"], errors="coerce")
    result["log_tumor_cells"] = np.log1p(result["n_tumor_cells"])
    result["log_mlh1_epithelial_cells"] = np.log1p(
        result["n_epithelial_cells_for_MLH1"]
    )
    return result


def validate_covariate_names(covariates: list[str] | tuple[str, ...]) -> None:
    """Reject identifiers, outcomes, and outcome-derived fields from adjustment."""

    if not covariates:
        raise ValueError("at least one covariate is required")
    duplicates = sorted({name for name in covariates if covariates.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate covariates: {duplicates}")
    forbidden = [
        name
        for name in covariates
        if any(token in str(name).lower() for token in FORBIDDEN_COVARIATE_TOKENS)
    ]
    if forbidden:
        raise ValueError(f"forbidden covariates: {forbidden}")


def _covariate_preprocessor(
    continuous_covariates: tuple[str, ...],
    binary_covariates: tuple[str, ...],
) -> ColumnTransformer:
    continuous = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )
    binary = Pipeline(
        [
            (
                "impute",
                SimpleImputer(strategy="most_frequent", keep_empty_features=True),
            )
        ]
    )
    return ColumnTransformer(
        [
            ("continuous", continuous, list(continuous_covariates)),
            ("binary", binary, list(binary_covariates)),
        ],
        remainder="drop",
    )


class MetadataResidualizer:
    """Remove the training-estimated linear metadata component from molecular features."""

    def __init__(
        self,
        alpha: float = 1.0,
        continuous_covariates: tuple[str, ...] = CONTINUOUS_COVARIATES,
        binary_covariates: tuple[str, ...] = BINARY_COVARIATES,
    ) -> None:
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be a finite non-negative number")
        self.alpha = float(alpha)
        self.continuous_covariates = tuple(continuous_covariates)
        self.binary_covariates = tuple(binary_covariates)
        self.covariates = self.continuous_covariates + self.binary_covariates
        validate_covariate_names(self.covariates)

    def _validate_metadata(self, metadata: pd.DataFrame) -> None:
        _require_columns(metadata, set(self.covariates), "adjustment metadata")

    @staticmethod
    def _molecular_frame(
        molecular: pd.DataFrame | np.ndarray,
        expected_columns: tuple[str, ...] | None = None,
    ) -> pd.DataFrame:
        if isinstance(molecular, pd.DataFrame):
            frame = molecular.copy()
        else:
            array = np.asarray(molecular, dtype=float)
            if array.ndim != 2:
                raise ValueError("molecular matrix must be two-dimensional")
            columns = (
                list(expected_columns)
                if expected_columns is not None
                else [f"feature_{index}" for index in range(array.shape[1])]
            )
            frame = pd.DataFrame(array, columns=columns)
        if expected_columns is not None:
            missing = sorted(set(expected_columns).difference(frame.columns))
            extra = sorted(set(frame.columns).difference(expected_columns))
            if missing or extra:
                raise ValueError(f"molecular columns differ; missing={missing}, extra={extra}")
            frame = frame.loc[:, list(expected_columns)]
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError("molecular matrix contains non-finite values")
        return numeric

    def fit(
        self,
        metadata: pd.DataFrame,
        molecular: pd.DataFrame | np.ndarray,
    ) -> "MetadataResidualizer":
        self._validate_metadata(metadata)
        molecular_frame = self._molecular_frame(molecular)
        if len(metadata) != len(molecular_frame):
            raise ValueError("metadata and molecular matrix row counts differ")
        self.feature_names_ = tuple(str(column) for column in molecular_frame.columns)
        self.preprocessor_ = _covariate_preprocessor(
            self.continuous_covariates, self.binary_covariates
        )
        covariate_matrix = self.preprocessor_.fit_transform(metadata)
        self.model_ = Ridge(alpha=self.alpha, fit_intercept=True)
        self.model_.fit(covariate_matrix, molecular_frame.to_numpy(dtype=float))
        return self

    def transform(
        self,
        metadata: pd.DataFrame,
        molecular: pd.DataFrame | np.ndarray,
    ) -> pd.DataFrame:
        if not hasattr(self, "model_"):
            raise ValueError("residualizer is not fitted")
        self._validate_metadata(metadata)
        molecular_frame = self._molecular_frame(molecular, self.feature_names_)
        if len(metadata) != len(molecular_frame):
            raise ValueError("metadata and molecular matrix row counts differ")
        covariate_matrix = self.preprocessor_.transform(metadata)
        predicted = np.asarray(self.model_.predict(covariate_matrix), dtype=float)
        if predicted.ndim == 1:
            predicted = predicted.reshape(-1, 1)
        if predicted.shape != molecular_frame.shape:
            raise ValueError(
                "residualizer prediction shape differs from the molecular matrix"
            )
        residuals = molecular_frame.to_numpy(dtype=float) - predicted
        return pd.DataFrame(
            residuals,
            index=molecular_frame.index,
            columns=list(self.feature_names_),
        )

    def fit_transform(
        self,
        metadata: pd.DataFrame,
        molecular: pd.DataFrame | np.ndarray,
    ) -> pd.DataFrame:
        return self.fit(metadata, molecular).transform(metadata, molecular)


def make_patient_grouped_folds(
    y: pd.Series | np.ndarray,
    patient_ids: pd.Series | np.ndarray,
    n_splits: int = 5,
    random_state: int = 2026,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return deterministic stratified folds with each patient kept intact."""

    labels = np.asarray(y, dtype=int)
    groups = np.asarray(patient_ids).astype(str)
    if labels.ndim != 1 or groups.ndim != 1 or len(labels) != len(groups):
        raise ValueError("labels and patient IDs must be aligned one-dimensional arrays")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("grouped folds require binary labels 0 and 1")
    label_frame = pd.DataFrame({"patientID": groups, "y": labels})
    conflicts = label_frame.groupby("patientID")["y"].nunique()
    if conflicts.gt(1).any():
        raise ValueError("a patient cannot have conflicting labels")
    splitter = StratifiedGroupKFold(
        n_splits=int(n_splits), shuffle=True, random_state=int(random_state)
    )
    return [
        (np.asarray(train_index, dtype=int), np.asarray(test_index, dtype=int))
        for train_index, test_index in splitter.split(
            np.zeros((len(labels), 1)), labels, groups
        )
    ]


def weighted_auc(
    y_true: pd.Series | np.ndarray,
    score: pd.Series | np.ndarray,
    sample_weight: pd.Series | np.ndarray,
) -> float:
    """Calculate the pair-weighted AUROC, assigning half credit to ties."""

    y = np.asarray(y_true, dtype=int)
    values = np.asarray(score, dtype=float)
    weights = np.asarray(sample_weight, dtype=float)
    if not (len(y) == len(values) == len(weights)):
        raise ValueError("weighted AUROC inputs must have equal lengths")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("weighted AUROC requires both classes")
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("weighted AUROC inputs must be finite with non-negative weights")
    positive_score = values[y == 1][:, None]
    negative_score = values[y == 0][None, :]
    pair_weight = weights[y == 1][:, None] * weights[y == 0][None, :]
    denominator = float(pair_weight.sum())
    if denominator <= 0:
        raise ValueError("each class must have positive total weight")
    credit = (positive_score > negative_score).astype(float)
    credit += 0.5 * (positive_score == negative_score)
    return float(np.sum(pair_weight * credit) / denominator)


def overlap_weights(
    metadata: pd.DataFrame,
    covariates: list[str] | tuple[str, ...],
    patient_weight: bool = True,
    random_state: int = 2026,
) -> pd.DataFrame:
    """Estimate overlap weights while giving each multi-region patient equal base mass."""

    covariates = list(covariates)
    validate_covariate_names(covariates)
    _require_columns(metadata, {"patientID", "y", *covariates}, "overlap metadata")
    result = metadata.loc[:, ["patientID", "y", *covariates]].copy()
    result["y"] = pd.to_numeric(result["y"], errors="raise").astype(int)
    if set(result["y"].unique()) != {0, 1}:
        raise ValueError("overlap weighting requires both classes")
    conflicts = result.groupby("patientID")["y"].nunique()
    if conflicts.gt(1).any():
        raise ValueError("a patient cannot have conflicting labels")
    if patient_weight:
        group_size = result.groupby("patientID")["patientID"].transform("size")
        base_weight = 1.0 / group_size.to_numpy(dtype=float)
    else:
        base_weight = np.ones(len(result), dtype=float)

    continuous = tuple(name for name in covariates if name in CONTINUOUS_COVARIATES)
    binary = tuple(name for name in covariates if name not in continuous)
    preprocessor = _covariate_preprocessor(continuous, binary)
    model = LogisticRegression(C=1.0, max_iter=10000, random_state=random_state)
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipeline.fit(
        result.loc[:, covariates],
        result["y"].to_numpy(),
        model__sample_weight=base_weight,
    )
    propensity = np.clip(
        pipeline.predict_proba(result.loc[:, covariates])[:, 1], 1e-6, 1.0 - 1e-6
    )
    overlap = np.where(result["y"].to_numpy() == 1, 1.0 - propensity, propensity)
    result["propensity"] = propensity
    result["base_patient_weight"] = base_weight
    result["overlap_weight"] = base_weight * overlap
    return result


def standardized_mean_difference(
    values: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    weights: pd.Series | np.ndarray | None = None,
) -> float:
    """Return the weighted standardized mean difference (MSI minus MSS)."""

    x = np.asarray(values, dtype=float)
    labels = np.asarray(y, dtype=int)
    sample_weight = np.ones(len(x), dtype=float) if weights is None else np.asarray(weights, float)
    if not (len(x) == len(labels) == len(sample_weight)):
        raise ValueError("standardized mean difference inputs must have equal lengths")

    moments = []
    for label in (0, 1):
        mask = labels == label
        valid = mask & np.isfinite(x) & np.isfinite(sample_weight)
        if not valid.any() or sample_weight[valid].sum() <= 0:
            raise ValueError(f"class {label} has no valid positive-weight observations")
        mean = float(np.average(x[valid], weights=sample_weight[valid]))
        variance = float(np.average((x[valid] - mean) ** 2, weights=sample_weight[valid]))
        moments.append((mean, variance))
    denominator = np.sqrt((moments[0][1] + moments[1][1]) / 2.0)
    if denominator == 0:
        return 0.0
    return float((moments[1][0] - moments[0][0]) / denominator)
