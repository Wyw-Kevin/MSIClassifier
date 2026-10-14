#!/usr/bin/env python3
"""Compare continuous and literature-grouped age adjustment without sidedness."""

from __future__ import annotations

# Packaging-only path resolution; archived originals retain exact source bytes.
from pathlib import Path as _MSIPackagePath
import sys as _MSIPackageSys
for _MSIPackageParent in _MSIPackagePath(__file__).resolve().parents:
    if (_MSIPackageParent / "project_paths.py").is_file():
        _MSIPackageSys.path.insert(0, str(_MSIPackageParent))
        break
from project_paths import project_root

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import run_scrna_clinical_ablation as reporting
import run_scrna_metadata_adjusted_pipeline as core
import scrna_clinical_ablation as ablation
from scrna_metadata_adjustment import (
    parse_gse132465_metadata,
    parse_gse178341_metadata,
)


FINAL = project_root() / '最终模型'
OUT = (
    FINAL
    / "data"
    / "scRNA"
    / "result"
    / "diagnostics"
    / "age_sex_no_sidedness_nested_20260915"
)
AGE_CUTOFF = 50
RANDOM_STATE = core.RANDOM_STATE
N_BOOTSTRAP = 2000
FEATURE_THRESHOLD = 0.65
MINIMUM_FEATURE_COUNT = 10
MAXIMUM_FEATURE_COUNT = 24
N_SPLITS = 5
PRIOR_OUT = (
    FINAL
    / "data"
    / "scRNA"
    / "result"
    / "diagnostics"
    / "clinical_adjustment_ablation_nested_20260915"
)
PRIOR_EXPRESSION_SCORES = PRIOR_OUT / "attainable_ablation_scores.csv"
TRACKS = {
    "age_continuous_sex": ablation.ClinicalTrack(("age",), ("female",)),
    "age_group50_sex": ablation.ClinicalTrack(
        (), ("early_onset_lt50", "female")
    ),
}
EXCLUDED_COVARIATES = {
    "right_sided",
    "t_high",
    "node_positive",
    "grade_high",
    "n_tumor_cells",
    "n_epithelial_cells_for_MLH1",
    "log_tumor_cells",
    "log_mlh1_epithelial_cells",
}
REQUIRED_CSV_OUTPUTS = (
    "fixed_shared18_AUROC.csv",
    "fixed_shared18_scores.csv",
    "rescreened_nested_AUROC.csv",
    "rescreened_nested_scores.csv",
    "rescreened_external_AUROC.csv",
    "rescreened_external_scores.csv",
    "paired_age_encoding_delta_AUROC.csv",
    "nested_fold_selections.csv",
    "development_locked_selections.csv",
    "feature_overlap.csv",
)


def add_literature_age_group(metadata: pd.DataFrame) -> pd.DataFrame:
    """Add the conventional early-onset CRC indicator without mutating input."""

    if "age" not in metadata.columns:
        raise ValueError("metadata is missing age")
    result = metadata.copy()
    age = pd.to_numeric(result["age"], errors="coerce")
    result["early_onset_lt50"] = np.where(
        age.notna(), age.lt(AGE_CUTOFF).astype(float), np.nan
    )
    return result


def validate_tracks() -> None:
    ablation.validate_track_contracts(TRACKS)
    for name, track in TRACKS.items():
        forbidden = sorted(set(track.covariates).intersection(EXCLUDED_COVARIATES))
        if forbidden:
            raise ValueError(f"{name} contains excluded covariates: {forbidden}")


def summarize_age_groups(metadata: pd.DataFrame) -> pd.DataFrame:
    required = {"dataset", "MMR", "early_onset_lt50"}
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise ValueError(f"metadata is missing age-summary columns: {missing}")
    frame = metadata.loc[:, ["dataset", "MMR", "early_onset_lt50"]].copy()
    frame["age_group"] = frame["early_onset_lt50"].map(
        {1.0: "<50", 0.0: ">=50"}
    )
    frame.loc[frame["early_onset_lt50"].isna(), "age_group"] = "missing"
    return (
        frame.groupby(["dataset", "age_group", "MMR"], sort=False, dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )


def _overlap_row(
    left_name: str,
    left_features: list[str],
    right_name: str,
    right_features: list[str],
) -> dict[str, object]:
    left = list(dict.fromkeys(left_features))
    right = list(dict.fromkeys(right_features))
    left_set = set(left)
    right_set = set(right)
    overlap = [feature for feature in left if feature in right_set]
    left_only = [feature for feature in left if feature not in right_set]
    right_only = [feature for feature in right if feature not in left_set]
    return {
        "comparison": f"{left_name}_vs_{right_name}",
        "left_panel": left_name,
        "right_panel": right_name,
        "n_left": len(left),
        "n_right": len(right),
        "n_overlap": len(overlap),
        "overlap_features": ";".join(overlap),
        "left_only_features": ";".join(left_only),
        "right_only_features": ";".join(right_only),
    }


def build_feature_overlap(
    selections: dict[str, list[str]], original_shared18: list[str]
) -> pd.DataFrame:
    missing = sorted(set(TRACKS).difference(selections))
    if missing:
        raise ValueError(f"missing development selections: {missing}")
    rows = [
        _overlap_row(
            track_name,
            list(selections[track_name]),
            "original_shared18",
            list(original_shared18),
        )
        for track_name in TRACKS
    ]
    rows.append(
        _overlap_row(
            "age_group50_sex",
            list(selections["age_group50_sex"]),
            "age_continuous_sex",
            list(selections["age_continuous_sex"]),
        )
    )
    return pd.DataFrame(rows)


def build_development_selection_table(
    selections: dict[str, ablation.CandidateSelection],
) -> pd.DataFrame:
    missing = sorted(set(TRACKS).difference(selections))
    if missing:
        raise ValueError(f"missing development selections: {missing}")
    return pd.DataFrame(
        [
            {
                "clinical_track": track_name,
                "configuration": "ecology_plus_MLH1",
                "selection_protocol": "development_grouped_oof_locked_external",
                "feature_set": selection.feature_set,
                "n_passing_screen": int(selection.n_passing_screen),
                "n_subtype_features": len(selection.subtype_features),
                "n_model_features": len(selection.subtype_features) + 1,
                "model": selection.model_name,
                "development_grouped_oof_auc": float(selection.inner_auc),
                "selected_features": ";".join(selection.subtype_features),
                "additional_feature": core.MLH1_FEATURE,
            }
            for track_name, selection in selections.items()
        ]
    )


def _summarize_track_scores(
    scores: pd.DataFrame,
    analysis_branch: str,
    train_split: str,
    notes: str,
    seed_start: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for index, ((track_name, split), part) in enumerate(
        scores.groupby(["clinical_track", "split"], sort=False)
    ):
        summary = reporting.summarize_scores(
            part,
            analysis=f"{analysis_branch}_{track_name}_{split}",
            include_patient_average=split == train_split,
            random_state=2026,
            notes=notes,
        )
        summary.insert(0, "analysis_branch", analysis_branch)
        summary.insert(1, "clinical_track", track_name)
        summary.insert(2, "split", split)
        rows.append(summary)
    if not rows:
        raise ValueError(f"no scores available for {analysis_branch}")
    return pd.concat(rows, ignore_index=True)


def _decorate_fixed_scores(
    scores: pd.DataFrame,
    track_name: str,
    track: ablation.ClinicalTrack,
) -> pd.DataFrame:
    result = scores.loc[
        scores["split"].isin(["train_grouped_oof", "external"])
    ].copy()
    result["analysis_branch"] = "fixed_shared18"
    result["clinical_track"] = track_name
    result["configuration"] = "fixed_shared18_plus_MLH1"
    result["selection_protocol"] = np.where(
        result["split"].eq("external"),
        "prespecified_development_fit_locked_external",
        "prespecified_patient_grouped_oof",
    )
    result["feature_set"] = "fixed_shared18_plus_mlh1"
    result["n_subtype_features"] = len(core.CURRENT_SHARED18)
    result["n_model_features"] = len(core.CURRENT_SHARED18) + 1
    result["model"] = "lda_shrinkage"
    result["selected_features"] = ";".join(core.CURRENT_SHARED18)
    result["additional_feature"] = core.MLH1_FEATURE
    result["adjustment_covariates"] = ";".join(track.covariates)
    return result


def run_fixed_shared18(
    train: pd.DataFrame,
    external: pd.DataFrame,
    metadata_train: pd.DataFrame,
    metadata_external: pd.DataFrame,
    outer_folds: list[tuple[np.ndarray, np.ndarray]],
    models: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_parts: list[pd.DataFrame] = []
    expression_track = ablation.ClinicalTrack((), ())
    _, expression_scores = core.evaluate_unadjusted_candidate(
        train=train,
        external=external,
        metadata_train=metadata_train,
        metadata_external=metadata_external,
        model_features=list(core.CURRENT_SHARED18) + [core.MLH1_FEATURE],
        model_template=models["lda_shrinkage"],
        model_name="lda_shrinkage",
        folds=outer_folds,
        random_state=RANDOM_STATE,
    )
    score_parts.append(
        _decorate_fixed_scores(expression_scores, "expression_only", expression_track)
    )

    for track_name, track in TRACKS.items():
        _, scores, _ = core.evaluate_adjusted_candidate(
            train=train,
            external=external,
            metadata_train=metadata_train,
            metadata_external=metadata_external,
            subtype_features=list(core.CURRENT_SHARED18),
            model_template=models["lda_shrinkage"],
            feature_set_name="fixed_shared18_plus_mlh1",
            model_name="lda_shrinkage",
            folds=outer_folds,
            random_state=RANDOM_STATE,
            alpha=core.NUISANCE_ALPHA,
            continuous_covariates=track.continuous_covariates,
            binary_covariates=track.binary_covariates,
            additional_features=(core.MLH1_FEATURE,),
        )
        score_parts.append(_decorate_fixed_scores(scores, track_name, track))

    combined = pd.concat(score_parts, ignore_index=True, sort=False)
    metrics = _summarize_track_scores(
        combined,
        analysis_branch="fixed_shared18",
        train_split="train_grouped_oof",
        notes="Prespecified shared18 plus MLH1 shrinkage LDA; no feature reselection",
        seed_start=2026,
    )
    return metrics, combined


def _validate_nested_result(
    result: ablation.NestedResult,
    train: pd.DataFrame,
    expected_folds: int,
    track_name: str,
) -> None:
    scores = result.scores
    if len(scores) != len(train) or scores["sampleID"].nunique() != len(train):
        raise ValueError(f"{track_name} nested scores do not contain one row per specimen")
    if not np.isfinite(scores["score"].to_numpy(dtype=float)).all():
        raise ValueError(f"{track_name} nested scores contain non-finite values")
    if sorted(scores["fold"].astype(int).unique().tolist()) != list(
        range(1, expected_folds + 1)
    ):
        raise ValueError(f"{track_name} nested scores do not cover all outer folds")
    if scores.groupby("patientID")["fold"].nunique().gt(1).any():
        raise ValueError(f"{track_name} assigns one patient to multiple outer folds")
    if len(result.selections) != expected_folds:
        raise ValueError(f"{track_name} does not have one selection per outer fold")
    if not result.failures.empty:
        raise ValueError(
            f"{track_name} nested selection failures: {result.failures.to_dict('records')}"
        )


def run_rescreened_analysis(
    train: pd.DataFrame,
    external: pd.DataFrame,
    metadata_train: pd.DataFrame,
    metadata_external: pd.DataFrame,
    candidates: list[str],
    outer_folds: list[tuple[np.ndarray, np.ndarray]],
    models: dict[str, object],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, ablation.CandidateSelection],
]:
    nested_score_parts: list[pd.DataFrame] = []
    nested_selection_parts: list[pd.DataFrame] = []
    development_selections: dict[str, ablation.CandidateSelection] = {}
    external_score_parts: list[pd.DataFrame] = []

    for index, (track_name, track) in enumerate(TRACKS.items(), start=1):
        print(f"[{index}/{len(TRACKS)}] strict nested re-screen: {track_name}", flush=True)
        nested = ablation.run_nested_configuration(
            train=train,
            metadata=metadata_train,
            candidate_features=candidates,
            models=models,
            outer_folds=outer_folds,
            track=track,
            configuration="ecology_plus_MLH1",
            threshold=FEATURE_THRESHOLD,
            minimum_count=MINIMUM_FEATURE_COUNT,
            maximum_count=MAXIMUM_FEATURE_COUNT,
            inner_splits=N_SPLITS,
            random_state=RANDOM_STATE,
        )
        _validate_nested_result(nested, train, len(outer_folds), track_name)
        nested_score_parts.append(
            nested.scores.assign(
                analysis_branch="rescreened_71",
                clinical_track=track_name,
            )
        )
        nested_selection_parts.append(
            nested.selections.assign(
                analysis_branch="rescreened_71",
                clinical_track=track_name,
            )
        )

        print(f"[{index}/{len(TRACKS)}] development lock: {track_name}", flush=True)
        selection = ablation.select_inner_candidate(
            train=train,
            metadata=metadata_train,
            candidate_features=candidates,
            models=models,
            track=track,
            configuration="ecology_plus_MLH1",
            threshold=FEATURE_THRESHOLD,
            minimum_count=MINIMUM_FEATURE_COUNT,
            maximum_count=MAXIMUM_FEATURE_COUNT,
            inner_splits=N_SPLITS,
            random_state=RANDOM_STATE,
        )
        if not selection.failures.empty:
            raise ValueError(
                f"{track_name} development-lock failures: "
                f"{selection.failures.to_dict('records')}"
            )
        development_selections[track_name] = selection
        external_score_parts.append(
            reporting._external_score_frame(
                train=train,
                external=external,
                metadata_train=metadata_train,
                metadata_external=metadata_external,
                models=models,
                track=track,
                track_name=track_name,
                configuration="ecology_plus_MLH1",
                selection=selection,
                random_state=2026,
            ).assign(analysis_branch="rescreened_71")
        )

    nested_scores = pd.concat(nested_score_parts, ignore_index=True, sort=False)
    nested_selections = pd.concat(
        nested_selection_parts, ignore_index=True, sort=False
    )
    external_scores = pd.concat(external_score_parts, ignore_index=True, sort=False)
    nested_metrics = _summarize_track_scores(
        nested_scores,
        analysis_branch="rescreened_71",
        train_split="train_nested_grouped_oof",
        notes="Complete outer-isolated 71-class screening and model selection",
        seed_start=2026,
    )
    external_metrics = _summarize_track_scores(
        external_scores,
        analysis_branch="rescreened_71_external",
        train_split="unused",
        notes="Development-locked panel and model; external outcomes unused for selection",
        seed_start=2026,
    )
    development_table = build_development_selection_table(development_selections)
    return (
        nested_metrics,
        nested_scores,
        external_metrics,
        external_scores,
        nested_selections,
        development_table,
        development_selections,
    )


def _score_subset(
    scores: pd.DataFrame, track_name: str, split: str
) -> pd.DataFrame:
    part = scores.loc[
        scores["clinical_track"].eq(track_name) & scores["split"].eq(split)
    ].copy()
    if part.empty:
        raise ValueError(f"missing score subset for {track_name}/{split}")
    return part


def _paired_comparison_row(
    first: pd.DataFrame,
    second: pd.DataFrame,
    analysis_branch: str,
    first_name: str,
    second_name: str,
    split: str,
    protocol: str,
    random_state: int,
) -> dict[str, object]:
    raw = reporting._paired_delta_row(
        first,
        second,
        track_name=first_name,
        comparison=f"{first_name}_minus_{second_name}",
        split=split,
        protocol=protocol,
        random_state=random_state,
    )
    return {
        "analysis_branch": analysis_branch,
        "split": raw["split"],
        "dataset": raw["dataset"],
        "comparison": raw["comparison"],
        "first_model": first_name,
        "second_model": second_name,
        "delta_definition": "first_model_auc_minus_second_model_auc",
        "comparison_protocol": raw["ablation_protocol"],
        "first_auc": raw["full_auc"],
        "second_auc": raw["ablated_auc"],
        "delta_auc": raw["delta_auc"],
        "delta_ci_lower": raw["delta_ci_lower"],
        "delta_ci_upper": raw["delta_ci_upper"],
        "bootstrap_success": raw["bootstrap_success"],
        "n": raw["n"],
        "n_patients": raw["n_patients"],
    }


def build_paired_comparisons(
    fixed_scores: pd.DataFrame,
    nested_scores: pd.DataFrame,
    external_scores: pd.DataFrame,
    prior_expression_scores: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    seed = 2026

    for offset, split in enumerate(("train_grouped_oof", "external")):
        rows.append(
            _paired_comparison_row(
                _score_subset(fixed_scores, "age_group50_sex", split),
                _score_subset(fixed_scores, "age_continuous_sex", split),
                "fixed_shared18",
                "age_group50_sex",
                "age_continuous_sex",
                split,
                "prespecified_fixed_panel_age_encoding",
                2026,
            )
        )

    score_sources = {
        "train_nested_grouped_oof": nested_scores,
        "external": external_scores,
    }
    for split_index, (split, source) in enumerate(score_sources.items()):
        rows.append(
            _paired_comparison_row(
                _score_subset(source, "age_group50_sex", split),
                _score_subset(source, "age_continuous_sex", split),
                "rescreened_71",
                "age_group50_sex",
                "age_continuous_sex",
                split,
                "independently_selected_age_encoding",
                2026,
            )
        )
        expression = _score_subset(prior_expression_scores, "expression_only", split)
        for track_index, track_name in enumerate(TRACKS):
            rows.append(
                _paired_comparison_row(
                    _score_subset(source, track_name, split),
                    expression,
                    "rescreened_71",
                    track_name,
                    "strict_expression_only",
                    split,
                    "independently_selected_adjusted_vs_read_only_expression_reference",
                    2026,
                )
            )
    return pd.DataFrame(rows)


def _sample_auc_row(
    metrics: pd.DataFrame, track_name: str, split: str
) -> pd.Series:
    rows = metrics.loc[
        metrics["clinical_track"].eq(track_name)
        & metrics["split"].eq(split)
        & metrics["estimand"].eq("sample_level_patient_cluster_bootstrap")
    ]
    if len(rows) != 1:
        raise ValueError(f"expected one sample-level AUROC for {track_name}/{split}")
    return rows.iloc[0]


def build_analysis_summary(
    fixed_metrics: pd.DataFrame,
    nested_metrics: pd.DataFrame,
    external_metrics: pd.DataFrame,
    development_table: pd.DataFrame,
    overlap: pd.DataFrame,
    paired: pd.DataFrame,
    age_counts: pd.DataFrame,
) -> str:
    labels = {
        "expression_only": "未校正",
        "age_continuous_sex": "连续年龄+性别",
        "age_group50_sex": "<50/\u226550岁分组+性别",
    }

    def formatted(row: pd.Series) -> str:
        return (
            f"{row['auc']:.3f} "
            f"({row['auc_ci_lower']:.3f}-{row['auc_ci_upper']:.3f})"
        )

    lines = [
        "# 仅校正年龄和性别的 scRNA 敏感性分析",
        "",
        "本分析不纳入肿瘤侧别，并将原始固定18亚型面板与71亚型重新筛选分开评估。以下均为 AUROC（95% 患者聚类 bootstrap CI）。",
        "",
        "## 固定18亚型 + MLH1",
        "",
        "| 处理 | GSE178341 患者分组 OOF | GSE132465 外部验证 |",
        "|---|---:|---:|",
    ]
    for track_name in ("expression_only", *TRACKS.keys()):
        train_row = _sample_auc_row(
            fixed_metrics, track_name, "train_grouped_oof"
        )
        external_row = _sample_auc_row(fixed_metrics, track_name, "external")
        lines.append(
            f"| {labels[track_name]} | {formatted(train_row)} | {formatted(external_row)} |"
        )

    lines.extend(
        [
            "",
            "该分支始终使用原18个细胞亚型、恶性上皮 MLH1 和 shrinkage LDA，因此性能变化只反映协变量处理，不反映特征重选。",
            "",
            "## 71亚型严格嵌套重筛",
            "",
            "| 处理 | GSE178341 严格嵌套 OOF | GSE132465 外部验证 |",
            "|---|---:|---:|",
        ]
    )
    for track_name in TRACKS:
        train_row = _sample_auc_row(
            nested_metrics, track_name, "train_nested_grouped_oof"
        )
        external_row = _sample_auc_row(external_metrics, track_name, "external")
        lines.append(
            f"| {labels[track_name]} | {formatted(train_row)} | {formatted(external_row)} |"
        )

    lines.extend(["", "开发队列锁定的特征和模型："])
    overlap_index = overlap.set_index(["left_panel", "right_panel"])
    for row in development_table.itertuples(index=False):
        overlap_row = overlap_index.loc[(row.clinical_track, "original_shared18")]
        lines.append(
            f"- {labels[row.clinical_track]}：{row.n_subtype_features}个亚型，"
            f"{row.model}，与原18亚型重叠{int(overlap_row['n_overlap'])}个；"
            f"特征为 {row.selected_features}。"
        )

    comparison = paired.loc[
        paired["analysis_branch"].eq("rescreened_71")
        & paired["split"].eq("train_nested_grouped_oof")
        & paired["comparison"].eq(
            "age_group50_sex_minus_age_continuous_sex"
        )
    ].iloc[0]
    lines.extend(
        [
            "",
            "## 解读边界",
            "",
            f"71亚型重筛中，年龄分组相对连续年龄的严格嵌套 OOF AUROC 差值为 "
            f"{comparison['delta_auc']:.3f} "
            f"({comparison['delta_ci_lower']:.3f}-{comparison['delta_ci_upper']:.3f})。",
            "年龄分组依据早发性结直肠癌常用的诊断年龄<50岁界值。开发队列仅10例<50岁（3 MSI、7 MSS），外部队列仅4例（2 MSI、2 MSS），因此分组年龄结果仅是敏感性分析。",
            "年龄二分会丢失组内差异，不能认为已完全消除年龄混杂。由于本分析明确排除侧别，也不能据此声称模型性能独立于肿瘤侧别。",
            "本次不替换现有主模型，不修改论文，不设定分类阈值，也不报告阈值依赖指标。",
            "",
        ]
    )
    return "\n".join(lines)


def protected_source_paths() -> dict[str, Path]:
    paths = {
        "GSE178341_features": core.TRAIN_FEATURES_PATH,
        "GSE132465_features": core.EXTERNAL_FEATURES_PATH,
        "unmerged71_contract": core.CLASS_CONTRACT_PATH,
        "GSE178341_metadata": core.GSE178341_METADATA_PATH,
        "GSE132465_metadata": core.GSE132465_METADATA_PATH,
        "current_model_predictions": core.CURRENT_PREDICTIONS_PATH,
        "current_scRNA_model": FINAL
        / "scRNA_model"
        / "models"
        / "scRNA_model"
        / "trained_msi_model.joblib",
        "current_scRNA_model_config": FINAL
        / "scRNA_model"
        / "models"
        / "scRNA_model"
        / "model_config.json",
        "current_scRNA_model_portable": FINAL
        / "scRNA_model"
        / "models"
        / "scRNA_model"
        / "trained_msi_model_portable.json",
        "current_manuscript": core.MANUSCRIPT_PATH,
    }
    for path in sorted(PRIOR_OUT.iterdir()):
        if path.is_file():
            paths[f"prior_diagnostic_{path.name}"] = path
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"protected sources are missing: {missing}")
    return paths


def build_manifest(
    source_paths: dict[str, Path],
    before_hashes: dict[str, str],
    after_hashes: dict[str, str],
    manuscript_mtime_before: int,
    manuscript_mtime_after: int,
    age_counts: pd.DataFrame,
    models: dict[str, object],
    nested_scores: pd.DataFrame,
    development_table: pd.DataFrame,
) -> dict[str, object]:
    age_records = age_counts.copy()
    age_records["n"] = age_records["n"].astype(int)
    return {
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "analysis": "scRNA age and sex adjustment without sidedness",
        "output_directory": str(OUT.resolve()),
        "primary_model_replaced": False,
        "manuscript_modified": manuscript_mtime_before != manuscript_mtime_after,
        "external_auc_used_for_selection": False,
        "protected_sources_unchanged": before_hashes == after_hashes,
        "clinical_tracks": {
            name: {
                "continuous_covariates": list(track.continuous_covariates),
                "binary_covariates": list(track.binary_covariates),
            }
            for name, track in TRACKS.items()
        },
        "age_definition": {
            "cutoff_years": AGE_CUTOFF,
            "early_onset_rule": "age < 50 years",
            "continuous_comparator": "age in years",
            "grouped_analysis_role": "sensitivity analysis",
        },
        "age_group_counts": age_records.to_dict("records"),
        "excluded_covariates": sorted(EXCLUDED_COVARIATES),
        "sidedness_included": False,
        "fixed_panel_contract": {
            "subtype_features": list(core.CURRENT_SHARED18),
            "additional_feature": core.MLH1_FEATURE,
            "model": "lda_shrinkage",
            "feature_reselection": False,
        },
        "rescreening_contract": {
            "candidate_subtypes": 71,
            "minimum_reference_cells": 50,
            "merged_subtypes": False,
            "screening_threshold": ">0.65 direction-corrected AUROC",
            "feature_prefix_range": "10 through min(24, n passing)",
            "selection_protocol": "outer-isolated patient-grouped nested selection",
        },
        "development_locked_selections": development_table.to_dict("records"),
        "outer_fold_assignment": nested_scores.loc[
            nested_scores["clinical_track"].eq("age_continuous_sex"),
            ["sampleID", "patientID", "fold"],
        ].drop_duplicates().to_dict("records"),
        "cohort_contracts": {
            "GSE178341": {"specimens": 64, "patients": 62, "MSI": 35, "MSS": 29},
            "GSE132465": {"specimens": 23, "patients": 23, "MSI": 4, "MSS": 19},
        },
        "random_state": RANDOM_STATE,
        "n_bootstrap": N_BOOTSTRAP,
        "n_outer_folds": N_SPLITS,
        "model_order": list(models),
        "failure_count": 0,
        "threshold_metrics_reported": False,
        "required_csv_outputs": list(REQUIRED_CSV_OUTPUTS),
        "references": {
            "early_onset_crc_boundary": "https://www.cancer.gov/about-nci/organization/dcb/news/early-onset-crc",
            "continuous_variable_dichotomization_caution": "https://pmc.ncbi.nlm.nih.gov/articles/PMC1458573/",
        },
        "source_files": {
            name: {"path": str(path.resolve()), "sha256": before_hashes[name]}
            for name, path in source_paths.items()
        },
        "read_only_hashes_before": before_hashes,
        "read_only_hashes_after": after_hashes,
        "manuscript_mtime_ns_before": int(manuscript_mtime_before),
        "manuscript_mtime_ns_after": int(manuscript_mtime_after),
    }


def write_outputs(
    output_dir: str | Path,
    tables: dict[str, pd.DataFrame],
    manifest: dict[str, object],
    summary: str,
) -> None:
    missing = sorted(set(REQUIRED_CSV_OUTPUTS).difference(tables))
    if missing:
        raise ValueError(f"missing required output tables: {missing}")
    for filename in REQUIRED_CSV_OUTPUTS:
        forbidden = sorted(
            set(tables[filename].columns).intersection(core.THRESHOLD_DERIVED_COLUMNS)
        )
        if forbidden:
            raise ValueError(
                f"{filename} contains threshold-derived columns: {forbidden}"
            )
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    for filename in REQUIRED_CSV_OUTPUTS:
        tables[filename].to_csv(target / filename, index=False)
    (target / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (target / "analysis_summary.md").write_text(summary, encoding="utf-8")


def main() -> None:
    validate_tracks()
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"non-overwrite output directory is not empty: {OUT}")

    train = pd.read_csv(core.TRAIN_FEATURES_PATH)
    external = pd.read_csv(core.EXTERNAL_FEATURES_PATH)
    contract = pd.read_csv(core.CLASS_CONTRACT_PATH)
    candidates = contract["class_name"].astype(str).tolist()
    metadata_train = parse_gse178341_metadata(core.GSE178341_METADATA_PATH)
    metadata_external = parse_gse132465_metadata(core.GSE132465_METADATA_PATH)
    metadata_train, metadata_external = core._validate_production_contract(
        train, external, metadata_train, metadata_external, candidates
    )
    metadata_train = add_literature_age_group(metadata_train)
    metadata_external = add_literature_age_group(metadata_external)
    age_counts = summarize_age_groups(
        pd.concat([metadata_train, metadata_external], ignore_index=True)
    )

    outer_folds = core.make_patient_grouped_folds(
        metadata_train["y"],
        metadata_train["patientID"],
        n_splits=N_SPLITS,
        random_state=RANDOM_STATE,
    )
    models = core.make_models(RANDOM_STATE)
    source_paths = protected_source_paths()
    before_hashes = reporting.snapshot_hashes(source_paths)
    manuscript_mtime_before = core.MANUSCRIPT_PATH.stat().st_mtime_ns

    print("[fixed] shared18 plus MLH1", flush=True)
    fixed_metrics, fixed_scores = run_fixed_shared18(
        train,
        external,
        metadata_train,
        metadata_external,
        outer_folds,
        models,
    )
    (
        nested_metrics,
        nested_scores,
        external_metrics,
        external_scores,
        nested_selections,
        development_table,
        development_selections,
    ) = run_rescreened_analysis(
        train,
        external,
        metadata_train,
        metadata_external,
        candidates,
        outer_folds,
        models,
    )

    prior_expression_scores = pd.read_csv(PRIOR_EXPRESSION_SCORES)
    prior_expression_scores = prior_expression_scores.loc[
        prior_expression_scores["clinical_track"].eq("expression_only")
        & prior_expression_scores["configuration"].eq("ecology_plus_MLH1")
        & prior_expression_scores["split"].isin(
            ["train_nested_grouped_oof", "external"]
        )
    ].copy()
    expected_prior_rows = len(train) + len(external)
    if len(prior_expression_scores) != expected_prior_rows:
        raise ValueError(
            "strict expression-only reference has an unexpected number of rows: "
            f"{len(prior_expression_scores)} != {expected_prior_rows}"
        )

    paired = build_paired_comparisons(
        fixed_scores,
        nested_scores,
        external_scores,
        prior_expression_scores,
    )
    selected_panels = {
        name: list(selection.subtype_features)
        for name, selection in development_selections.items()
    }
    overlap = build_feature_overlap(selected_panels, list(core.CURRENT_SHARED18))
    summary = build_analysis_summary(
        fixed_metrics,
        nested_metrics,
        external_metrics,
        development_table,
        overlap,
        paired,
        age_counts,
    )

    after_hashes = reporting.snapshot_hashes(source_paths)
    manuscript_mtime_after = core.MANUSCRIPT_PATH.stat().st_mtime_ns
    if before_hashes != after_hashes:
        raise RuntimeError("a protected source changed during analysis")
    if manuscript_mtime_before != manuscript_mtime_after:
        raise RuntimeError("the manuscript timestamp changed during analysis")

    manifest = build_manifest(
        source_paths,
        before_hashes,
        after_hashes,
        manuscript_mtime_before,
        manuscript_mtime_after,
        age_counts,
        models,
        nested_scores,
        development_table,
    )
    tables = {
        "fixed_shared18_AUROC.csv": fixed_metrics,
        "fixed_shared18_scores.csv": fixed_scores,
        "rescreened_nested_AUROC.csv": nested_metrics,
        "rescreened_nested_scores.csv": nested_scores,
        "rescreened_external_AUROC.csv": external_metrics,
        "rescreened_external_scores.csv": external_scores,
        "paired_age_encoding_delta_AUROC.csv": paired,
        "nested_fold_selections.csv": nested_selections,
        "development_locked_selections.csv": development_table,
        "feature_overlap.csv": overlap,
    }
    write_outputs(OUT, tables, manifest, summary)
    print(f"Output: {OUT}", flush=True)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
