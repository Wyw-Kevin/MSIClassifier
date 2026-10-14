#!/usr/bin/env python3
"""Run leakage-isolated clinical adjustment and ablation analyses for scRNA MSI."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import run_scrna_metadata_adjusted_pipeline as core
import scrna_clinical_ablation as ablation
from scrna_metadata_adjustment import (
    parse_gse132465_metadata,
    parse_gse178341_metadata,
)


FINAL = Path(__file__).resolve().parents[1]
OUT = (
    FINAL
    / "data"
    / "scRNA"
    / "result"
    / "diagnostics"
    / "clinical_adjustment_ablation_nested_20260915"
)
RANDOM_STATE = core.RANDOM_STATE
N_BOOTSTRAP = 2000
SOURCE_RUNNER = "run_scrna_clinical_ablation"
FEATURE_THRESHOLD = 0.65
MINIMUM_FEATURE_COUNT = 10
MAXIMUM_FEATURE_COUNT = 24
N_SPLITS = 5
CLINICAL_TRACKS = ablation.CLINICAL_TRACKS
REQUIRED_CSV_OUTPUTS = (
    "clinical_track_nested_AUROC.csv",
    "clinical_track_nested_scores.csv",
    "clinical_only_baseline_AUROC.csv",
    "clinical_only_baseline_scores.csv",
    "attainable_ablation_AUROC.csv",
    "attainable_ablation_scores.csv",
    "paired_ablation_delta_AUROC.csv",
    "conditional_lineage_ablation_AUROC.csv",
    "conditional_lineage_ablation_scores.csv",
    "exploratory_leave_one_feature_out_AUROC.csv",
    "exploratory_leave_one_feature_out_scores.csv",
    "external_ablation_AUROC.csv",
    "nested_fold_selections.csv",
    "nested_inner_candidate_AUROC.csv",
    "paired_ablation_bootstrap_replicates.csv.gz",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_hashes(source_paths: dict[str, str | Path]) -> dict[str, str]:
    return {name: sha256_file(path) for name, path in source_paths.items()}


def write_output_table(table: pd.DataFrame, path: str | Path) -> None:
    target = Path(path)
    if target.suffix == ".gz":
        table.to_csv(
            target,
            index=False,
            compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
        )
    else:
        table.to_csv(target, index=False)


def build_manifest(
    output_dir: str | Path,
    source_paths: dict[str, str | Path],
    read_only_before: dict[str, str],
    read_only_after: dict[str, str],
    manuscript_mtime_before: int,
    manuscript_mtime_after: int,
    model_order: list[str],
    failure_count: int,
) -> dict[str, object]:
    source_files = {
        name: {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}
        for name, path in source_paths.items()
    }
    tracks = {
        name: list(track.covariates) for name, track in CLINICAL_TRACKS.items()
    }
    return {
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "analysis": "scRNA clinical-adjustment and nested component ablation",
        "output_directory": str(Path(output_dir).resolve()),
        "primary_model_replaced": False,
        "manuscript_modified": manuscript_mtime_before != manuscript_mtime_after,
        "external_auc_used_for_selection": False,
        "clinical_tracks": tracks,
        "excluded_adjustment_domains": [
            "tumor and epithelial cell yield",
            "sample and patient identifiers",
            "MSI/MMR-derived labels",
            "cohort-specific technical fields",
        ],
        "selection_protocol": "outer-isolated patient-grouped nested model selection",
        "ablation_protocols": {
            "attainable": "each input configuration selected independently inside outer training",
            "conditional": "full-model fold selection locked before component removal",
            "feature_level": "development-locked exploratory leave-one-subtype-out",
        },
        "random_state": RANDOM_STATE,
        "n_bootstrap": N_BOOTSTRAP,
        "model_order": list(model_order),
        "failure_count": int(failure_count),
        "source_files": source_files,
        "read_only_hashes_before": dict(read_only_before),
        "read_only_hashes_after": dict(read_only_after),
        "read_only_sources_unchanged": read_only_before == read_only_after,
        "manuscript_mtime_ns_before": int(manuscript_mtime_before),
        "manuscript_mtime_ns_after": int(manuscript_mtime_after),
        "external_limitation": (
            "GSE132465 contains four MSI tumors, all right-sided; external ablation "
            "comparisons are descriptive."
        ),
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
        write_output_table(tables[filename], target / filename)
    (target / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (target / "analysis_summary.md").write_text(summary, encoding="utf-8")


def summarize_scores(
    scores: pd.DataFrame,
    analysis: str,
    include_patient_average: bool,
    random_state: int,
    notes: str,
) -> pd.DataFrame:
    required = {"dataset", "sampleID", "patientID", "y", "score"}
    missing = sorted(required.difference(scores.columns))
    if missing:
        raise ValueError(f"score table is missing columns: {missing}")
    if scores["dataset"].nunique() != 1:
        raise ValueError("one AUROC summary cannot combine datasets")
    rows: list[dict[str, object]] = []

    def append_row(frame: pd.DataFrame, estimand: str, seed: int) -> None:
        auc, lower, upper, success = core.stratified_bootstrap_auc(
            frame["y"],
            frame["score"],
            groups=frame["patientID"],
            n_bootstrap=N_BOOTSTRAP,
            random_state=seed,
        )
        rows.append(
            {
                "analysis": analysis,
                "dataset": str(scores["dataset"].iloc[0]),
                "estimand": estimand,
                "n": int(len(frame)),
                "n_patients": int(frame["patientID"].nunique()),
                "MSI": int(frame["y"].eq(1).sum()),
                "MSS": int(frame["y"].eq(0).sum()),
                "auc": auc,
                "auc_ci_lower": lower,
                "auc_ci_upper": upper,
                "bootstrap_success": success,
                "notes": notes,
            }
        )

    append_row(scores, "sample_level_patient_cluster_bootstrap", random_state)
    if include_patient_average and scores["patientID"].nunique() < len(scores):
        averaged = ablation.patient_average_scores(
            scores["y"], scores["score"], scores["patientID"]
        )
        append_row(
            averaged.assign(dataset=str(scores["dataset"].iloc[0])),
            "patient_average_auc",
            2026,
        )
    return pd.DataFrame(rows)


def _external_score_frame(
    train: pd.DataFrame,
    external: pd.DataFrame,
    metadata_train: pd.DataFrame,
    metadata_external: pd.DataFrame,
    models: dict[str, object],
    track: ablation.ClinicalTrack,
    track_name: str,
    configuration: str,
    selection: ablation.CandidateSelection,
    random_state: int,
) -> pd.DataFrame:
    additional = (
        (core.MLH1_FEATURE,) if configuration == "ecology_plus_MLH1" else ()
    )
    model_features = list(selection.subtype_features) + list(additional)
    score = ablation._fit_selected_and_score(
        train=train,
        test=external,
        metadata_train=metadata_train,
        metadata_test=metadata_external,
        model_features=model_features,
        model_template=models[selection.model_name],
        track=track,
        random_state=random_state,
    )
    aligned = core._align_metadata(external, metadata_external, "external")
    return pd.DataFrame(
        {
            "split": "external",
            "dataset": "GSE132465",
            "clinical_track": track_name,
            "configuration": configuration,
            "selection_protocol": "development_grouped_oof_locked_external",
            "sampleID": external["sampleID"].astype(str).to_numpy(),
            "patientID": aligned["patientID"].astype(str).to_numpy(),
            "MMR": external["MMR"].astype(str).to_numpy(),
            "y": core._binary_labels(external, "external"),
            "fold": 0,
            "score": score,
            "selected_feature_set": selection.feature_set,
            "n_subtype_features": len(selection.subtype_features),
            "n_model_features": len(model_features),
            "selected_model": selection.model_name,
            "selected_features": ";".join(selection.subtype_features),
        }
    )


def _external_mlh1_score_frame(
    train: pd.DataFrame,
    external: pd.DataFrame,
    metadata_train: pd.DataFrame,
    metadata_external: pd.DataFrame,
    track: ablation.ClinicalTrack,
    track_name: str,
) -> pd.DataFrame:
    score, direction_sign = ablation.score_external_mlh1_only(
        train,
        external,
        metadata_train,
        metadata_external,
        track,
    )
    aligned = core._align_metadata(external, metadata_external, "external")
    return pd.DataFrame(
        {
            "split": "external",
            "dataset": "GSE132465",
            "clinical_track": track_name,
            "configuration": "MLH1_only",
            "selection_protocol": "development_oriented_MLH1_external",
            "sampleID": external["sampleID"].astype(str).to_numpy(),
            "patientID": aligned["patientID"].astype(str).to_numpy(),
            "MMR": external["MMR"].astype(str).to_numpy(),
            "y": core._binary_labels(external, "external"),
            "fold": 0,
            "score": score,
            "selected_feature_set": "MLH1_only",
            "n_subtype_features": 0,
            "n_model_features": 1,
            "selected_model": "directional_MLH1",
            "selected_features": "",
            "direction_sign": direction_sign,
        }
    )


def _aligned_pair(
    full: pd.DataFrame, comparator: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    full_sorted = full.sort_values("sampleID", kind="mergesort").reset_index(drop=True)
    comparator_sorted = comparator.sort_values("sampleID", kind="mergesort").reset_index(
        drop=True
    )
    for column in ("sampleID", "patientID", "y"):
        if not np.array_equal(full_sorted[column], comparator_sorted[column]):
            raise ValueError(f"paired score tables differ in {column}")
    if "fold" in full_sorted and "fold" in comparator_sorted:
        if not np.array_equal(full_sorted["fold"], comparator_sorted["fold"]):
            raise ValueError("paired score tables differ in outer-fold assignment")
    return full_sorted, comparator_sorted


def _attach_bootstrap_context(
    replicates: pd.DataFrame,
    summaries: pd.DataFrame,
    source_runner: str,
) -> pd.DataFrame:
    context_keys = ["clinical_track", "comparison", "split"]
    summary_columns = context_keys + ["dataset", "ablation_protocol"]
    missing_replicates = sorted(
        set(context_keys + ["replicate"]).difference(replicates.columns)
    )
    missing_summary = sorted(set(summary_columns).difference(summaries.columns))
    if missing_replicates or missing_summary:
        raise ValueError(
            "bootstrap context inputs are incomplete; "
            f"replicates={missing_replicates}, summaries={missing_summary}"
        )
    if summaries.duplicated(context_keys).any():
        raise ValueError("paired summaries must be unique by track/comparison/split")
    enriched = replicates.merge(
        summaries.loc[:, summary_columns],
        on=context_keys,
        how="left",
        validate="many_to_one",
    )
    if enriched[["dataset", "ablation_protocol"]].isna().any().any():
        raise ValueError("a bootstrap replicate group has no paired summary context")
    enriched.insert(0, "source_runner", str(source_runner))
    ordered = [
        "source_runner",
        "clinical_track",
        "comparison",
        "split",
        "dataset",
        "ablation_protocol",
        "replicate",
    ]
    remainder = [column for column in enriched.columns if column not in ordered]
    return enriched.loc[:, ordered + remainder]


def _validate_bootstrap_replicates(
    replicates: pd.DataFrame,
    expected_replicates: int,
) -> None:
    group_keys = [
        "source_runner",
        "clinical_track",
        "comparison",
        "split",
    ]
    unique_keys = group_keys + ["replicate"]
    missing = sorted(set(unique_keys).difference(replicates.columns))
    if missing:
        raise ValueError(f"bootstrap replicate audit is missing columns: {missing}")
    if replicates.duplicated(unique_keys).any():
        raise ValueError("bootstrap replicate audit contains duplicate replicate keys")
    group_sizes = replicates.groupby(group_keys, sort=False).size()
    if group_sizes.empty or not group_sizes.eq(int(expected_replicates)).all():
        raise ValueError(
            f"each bootstrap context must contain {int(expected_replicates)} replicates"
        )


def _paired_delta_row(
    full: pd.DataFrame,
    comparator: pd.DataFrame,
    track_name: str,
    comparison: str,
    split: str,
    protocol: str,
    random_state: int,
    source_runner: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    full_aligned, comparator_aligned = _aligned_pair(full, comparator)
    result, replicates = ablation.paired_patient_bootstrap_auc_delta_with_replicates(
        full_aligned["y"],
        full_aligned["score"],
        comparator_aligned["score"],
        full_aligned["patientID"],
        n_bootstrap=N_BOOTSTRAP,
        random_state=random_state,
    )
    summary = {
        "clinical_track": track_name,
        "split": split,
        "dataset": str(full_aligned["dataset"].iloc[0]),
        "comparison": comparison,
        "ablation_protocol": protocol,
        **result,
    }
    replicates.insert(0, "clinical_track", track_name)
    replicates.insert(1, "comparison", comparison)
    replicates.insert(2, "split", split)
    replicates["random_state"] = int(random_state)
    replicates = _attach_bootstrap_context(
        replicates,
        pd.DataFrame([summary]),
        source_runner=source_runner,
    )
    return summary, replicates


def _summaries_by_configuration(
    scores: pd.DataFrame,
    split: str,
    include_patient_average: bool,
    seed_start: int,
    notes: str,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    subset = scores.loc[scores["split"].eq(split)]
    for index, ((track_name, configuration), part) in enumerate(
        subset.groupby(["clinical_track", "configuration"], sort=False)
    ):
        summary = summarize_scores(
            part,
            analysis=f"{track_name}_{configuration}_{split}",
            include_patient_average=include_patient_average,
            random_state=2026,
            notes=notes,
        )
        summary.insert(0, "clinical_track", track_name)
        summary.insert(1, "configuration", configuration)
        summary.insert(2, "split", split)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def _summarize_conditional_scores(
    scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_parts: list[pd.DataFrame] = []
    delta_rows: list[dict[str, object]] = []
    replicate_parts: list[pd.DataFrame] = []
    for track_index, (track_name, track_scores) in enumerate(
        scores.groupby("clinical_track", sort=False)
    ):
        for ablation_index, (ablation_name, part) in enumerate(
            track_scores.groupby("ablation", sort=False)
        ):
            summary = summarize_scores(
                part,
                analysis=f"{track_name}_{ablation_name}_conditional",
                include_patient_average=True,
                random_state=2026,
                notes="Full outer-fold selection locked before component removal",
            )
            fold_flags = part.loc[:, ["fold", "ablation_applied"]].drop_duplicates()
            summary.insert(0, "clinical_track", track_name)
            summary.insert(1, "ablation", ablation_name)
            summary.insert(2, "selection_protocol", "conditional_outer_selected_full_model")
            summary["n_outer_folds_applied"] = int(fold_flags["ablation_applied"].sum())
            summary["n_outer_folds"] = int(fold_flags["fold"].nunique())
            summary["outer_fold_application_fraction"] = (
                summary["n_outer_folds_applied"] / summary["n_outer_folds"]
            )
            metric_parts.append(summary)
            if ablation_name != "full":
                comparator = part.copy()
                full = part.copy()
                full["score"] = full["full_score"]
                delta, replicates = _paired_delta_row(
                    full,
                    comparator,
                    track_name,
                    f"full_minus_{ablation_name}",
                    "train_nested_grouped_oof",
                    "conditional_outer_selected_full_model",
                    2026,
                    source_runner=SOURCE_RUNNER,
                )
                delta_rows.append(delta)
                replicate_parts.append(replicates)
    return (
        pd.concat(metric_parts, ignore_index=True),
        pd.DataFrame(delta_rows),
        pd.concat(replicate_parts, ignore_index=True),
    )


def _summarize_leave_one_out(scores: pd.DataFrame) -> pd.DataFrame:
    metric_parts: list[pd.DataFrame] = []
    filtered = scores.loc[scores["split"].isin(["train_grouped_oof", "external"])]
    for index, ((track_name, ablation_name, removed_feature, split), part) in enumerate(
        filtered.groupby(
            ["clinical_track", "ablation", "removed_feature", "split"],
            sort=False,
            dropna=False,
        )
    ):
        summary = summarize_scores(
            part,
            analysis=f"{track_name}_{ablation_name}_{split}_exploratory",
            include_patient_average=False,
            random_state=2026,
            notes="Development-locked panel; exploratory leave-one-subtype-out",
        )
        summary.insert(0, "clinical_track", track_name)
        summary.insert(1, "ablation", ablation_name)
        summary.insert(2, "removed_feature", removed_feature)
        summary.insert(3, "split", split)
        metric_parts.append(summary)
    return pd.concat(metric_parts, ignore_index=True)


def build_analysis_summary(
    attainable_metrics: pd.DataFrame,
    paired_deltas: pd.DataFrame,
    external_metrics: pd.DataFrame,
    clinical_metrics: pd.DataFrame,
) -> str:
    sample_estimand = "sample_level_patient_cluster_bootstrap"

    def auc_for(table: pd.DataFrame, track: str, configuration: str) -> pd.Series:
        row = table.loc[
            table["clinical_track"].eq(track)
            & table["configuration"].eq(configuration)
            & table["estimand"].eq(sample_estimand)
        ]
        if len(row) != 1:
            raise ValueError(f"missing unique AUROC row for {track}/{configuration}")
        return row.iloc[0]

    lines = [
        "# 单细胞临床校正与消融分析",
        "",
        "## 完全隔离模型选择后的性能",
        "",
        "特征筛选、特征数量和分类器选择均在外层训练患者内部完成。",
        "",
    ]
    labels = {
        "expression_only": "未进行临床残差化",
        "core_age_sex_side": "年龄、性别和肿瘤侧别校正",
        "expanded_clinical6": "六项临床变量扩展校正",
    }
    for track_name in CLINICAL_TRACKS:
        row = auc_for(attainable_metrics, track_name, "ecology_plus_MLH1")
        lines.append(
            f"- {labels[track_name]}：完整模型 AUROC {row['auc']:.3f} "
            f"（95% CI {row['auc_ci_lower']:.3f}-{row['auc_ci_upper']:.3f}）。"
        )
    lines.extend(["", "## 细胞生态的增量价值", ""])
    development_delta = paired_deltas.loc[
        paired_deltas["split"].eq("train_nested_grouped_oof")
        & paired_deltas["ablation_protocol"].eq("attainable_independent_selection")
    ]
    for track_name in CLINICAL_TRACKS:
        full = auc_for(attainable_metrics, track_name, "ecology_plus_MLH1")
        mlh1 = auc_for(attainable_metrics, track_name, "MLH1_only")
        delta = development_delta.loc[
            development_delta["clinical_track"].eq(track_name)
            & development_delta["comparison"].eq("full_minus_MLH1_only")
        ].iloc[0]
        lines.append(
            f"- {labels[track_name]}：去掉细胞生态后 AUROC 从 {full['auc']:.3f} "
            f"变为 {mlh1['auc']:.3f}，配对 delta AUROC={delta['delta_auc']:.3f} "
            f"（95% CI {delta['delta_ci_lower']:.3f}-{delta['delta_ci_upper']:.3f}）。"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "核心校正用于检验年龄、性别和肿瘤侧别是否解释模型；六变量扩展校正还加入 T 分组、淋巴结状态和病理分级，可能同时去除部分 MSI 相关下游生物学。总肿瘤细胞数和 MLH1 上皮细胞数量未进入任何校正集合。",
            "",
            "GSE132465 只有 4 例 MSI，且全部为右侧肿瘤；外部 AUROC 与消融差值仅作描述性证据。当前未设分类阈值，因此使用价值仅解释为增量判别能力和非 BAM 转录组输入能力，不推断净获益、敏感度或特异度。",
            "",
            "## 临床变量基线",
            "",
        ]
    )
    for track_name in ("core_age_sex_side", "expanded_clinical6"):
        row = clinical_metrics.loc[
            clinical_metrics["clinical_track"].eq(track_name)
            & clinical_metrics["split"].eq("train_grouped_oof")
            & clinical_metrics["estimand"].eq(sample_estimand)
        ].iloc[0]
        lines.append(f"- {labels[track_name]}临床变量基线 AUROC：{row['auc']:.3f}。")
    lines.extend(["", "## 外部描述性结果", ""])
    for track_name in CLINICAL_TRACKS:
        row = auc_for(external_metrics, track_name, "ecology_plus_MLH1")
        lines.append(f"- {labels[track_name]}完整模型外部 AUROC：{row['auc']:.3f}。")
    return "\n".join(lines) + "\n"


def main() -> None:
    train = pd.read_csv(core.TRAIN_FEATURES_PATH)
    external = pd.read_csv(core.EXTERNAL_FEATURES_PATH)
    contract = pd.read_csv(core.CLASS_CONTRACT_PATH)
    candidates = contract["class_name"].astype(str).tolist()
    class_to_group = dict(
        zip(contract["class_name"].astype(str), contract["group_class"].astype(str))
    )
    ablation.validate_track_contracts()
    ablation.validate_lineage_contract(candidates, class_to_group)

    metadata_train = parse_gse178341_metadata(core.GSE178341_METADATA_PATH)
    metadata_external = parse_gse132465_metadata(core.GSE132465_METADATA_PATH)
    metadata_train, metadata_external = core._validate_production_contract(
        train, external, metadata_train, metadata_external, candidates
    )
    outer_folds = core.make_patient_grouped_folds(
        metadata_train["y"],
        metadata_train["patientID"],
        n_splits=N_SPLITS,
        random_state=RANDOM_STATE,
    )
    models = core.make_models(RANDOM_STATE)

    source_paths = {
        "GSE178341_features": core.TRAIN_FEATURES_PATH,
        "GSE132465_features": core.EXTERNAL_FEATURES_PATH,
        "unmerged71_contract": core.CLASS_CONTRACT_PATH,
        "GSE178341_metadata": core.GSE178341_METADATA_PATH,
        "GSE132465_metadata": core.GSE132465_METADATA_PATH,
        "current_model_predictions": core.CURRENT_PREDICTIONS_PATH,
        "current_scRNA_model": FINAL / "scRNA_model" / "models" / "scRNA_model" / "trained_msi_model.joblib",
        "current_manuscript": core.MANUSCRIPT_PATH,
    }
    before_hashes = snapshot_hashes(source_paths)
    manuscript_mtime_before = core.MANUSCRIPT_PATH.stat().st_mtime_ns

    nested_results: dict[tuple[str, str], ablation.NestedResult] = {}
    nested_score_parts: list[pd.DataFrame] = []
    nested_selection_parts: list[pd.DataFrame] = []
    nested_candidate_parts: list[pd.DataFrame] = []
    failure_parts: list[pd.DataFrame] = []
    for track_index, (track_name, track) in enumerate(CLINICAL_TRACKS.items(), start=1):
        for configuration in ("ecology_plus_MLH1", "ecology_only"):
            print(f"[{track_index}/3] nested {track_name}: {configuration}", flush=True)
            result = ablation.run_nested_configuration(
                train=train,
                metadata=metadata_train,
                candidate_features=candidates,
                models=models,
                outer_folds=outer_folds,
                track=track,
                configuration=configuration,
                threshold=FEATURE_THRESHOLD,
                minimum_count=MINIMUM_FEATURE_COUNT,
                maximum_count=MAXIMUM_FEATURE_COUNT,
                inner_splits=N_SPLITS,
                random_state=RANDOM_STATE,
            )
            nested_results[(track_name, configuration)] = result
            if not np.isfinite(result.scores["score"]).all():
                raise ValueError(f"incomplete nested scores for {track_name}/{configuration}")
            nested_score_parts.append(result.scores.assign(clinical_track=track_name))
            nested_selection_parts.append(
                result.selections.assign(clinical_track=track_name)
            )
            nested_candidate_parts.append(
                result.inner_candidates.assign(clinical_track=track_name)
            )
            if not result.failures.empty:
                failure_parts.append(result.failures.assign(clinical_track=track_name))
        print(f"[{track_index}/3] nested {track_name}: MLH1_only", flush=True)
        mlh1_result = ablation.run_nested_mlh1_only(
            train,
            metadata_train,
            outer_folds,
            track,
            RANDOM_STATE,
        )
        nested_results[(track_name, "MLH1_only")] = mlh1_result
        if not np.isfinite(mlh1_result.scores["score"]).all():
            raise ValueError(f"incomplete nested MLH1 scores for {track_name}")
        nested_score_parts.append(mlh1_result.scores.assign(clinical_track=track_name))
        nested_selection_parts.append(
            mlh1_result.selections.assign(clinical_track=track_name)
        )
        nested_candidate_parts.append(
            mlh1_result.inner_candidates.assign(clinical_track=track_name)
        )
        if not mlh1_result.failures.empty:
            failure_parts.append(mlh1_result.failures.assign(clinical_track=track_name))

    nested_scores = pd.concat(nested_score_parts, ignore_index=True)
    nested_selections = pd.concat(nested_selection_parts, ignore_index=True)
    nested_candidates = pd.concat(nested_candidate_parts, ignore_index=True)
    attainable_metrics = _summaries_by_configuration(
        nested_scores,
        split="train_nested_grouped_oof",
        include_patient_average=True,
        seed_start=2026,
        notes="Complete outer-isolated feature and model selection",
    )

    attainable_delta_rows: list[dict[str, object]] = []
    paired_replicate_parts: list[pd.DataFrame] = []
    for track_index, track_name in enumerate(CLINICAL_TRACKS):
        full = nested_scores.loc[
            nested_scores["clinical_track"].eq(track_name)
            & nested_scores["configuration"].eq("ecology_plus_MLH1")
        ]
        for comparison_index, comparator_name in enumerate(
            ("ecology_only", "MLH1_only")
        ):
            comparator = nested_scores.loc[
                nested_scores["clinical_track"].eq(track_name)
                & nested_scores["configuration"].eq(comparator_name)
            ]
            delta, replicates = _paired_delta_row(
                full,
                comparator,
                track_name,
                f"full_minus_{comparator_name}",
                "train_nested_grouped_oof",
                "attainable_independent_selection",
                2026,
                source_runner=SOURCE_RUNNER,
            )
            attainable_delta_rows.append(delta)
            paired_replicate_parts.append(replicates)

    development_selections: dict[tuple[str, str], ablation.CandidateSelection] = {}
    external_score_parts: list[pd.DataFrame] = []
    leave_one_score_parts: list[pd.DataFrame] = []
    conditional_score_parts: list[pd.DataFrame] = []
    for track_index, (track_name, track) in enumerate(CLINICAL_TRACKS.items(), start=1):
        print(f"[{track_index}/3] development lock and ablation: {track_name}", flush=True)
        for configuration_index, configuration in enumerate(
            ("ecology_plus_MLH1", "ecology_only")
        ):
            selection = ablation.select_inner_candidate(
                train=train,
                metadata=metadata_train,
                candidate_features=candidates,
                models=models,
                track=track,
                configuration=configuration,
                threshold=FEATURE_THRESHOLD,
                minimum_count=MINIMUM_FEATURE_COUNT,
                maximum_count=MAXIMUM_FEATURE_COUNT,
                inner_splits=N_SPLITS,
                random_state=RANDOM_STATE,
            )
            development_selections[(track_name, configuration)] = selection
            if not selection.failures.empty:
                failure_parts.append(
                    selection.failures.assign(
                        clinical_track=track_name,
                        outer_fold=0,
                        configuration=configuration,
                    )
                )
            external_score_parts.append(
                _external_score_frame(
                    train,
                    external,
                    metadata_train,
                    metadata_external,
                    models,
                    track,
                    track_name,
                    configuration,
                    selection,
                    2026,
                )
            )
        external_score_parts.append(
            _external_mlh1_score_frame(
                train,
                external,
                metadata_train,
                metadata_external,
                track,
                track_name,
            )
        )
        full_selection = development_selections[(track_name, "ecology_plus_MLH1")]
        _, leave_scores = ablation.run_locked_leave_one_feature_out(
            train=train,
            external=external,
            metadata_train=metadata_train,
            metadata_external=metadata_external,
            selected_features=list(full_selection.subtype_features),
            selected_model_name=full_selection.model_name,
            models=models,
            folds=outer_folds,
            track=track,
        )
        leave_one_score_parts.append(leave_scores.assign(clinical_track=track_name))
        conditional_score_parts.append(
            ablation.run_conditional_ablation(
                train=train,
                metadata=metadata_train,
                outer_folds=outer_folds,
                full_selections=nested_results[
                    (track_name, "ecology_plus_MLH1")
                ].selections,
                models=models,
                class_to_group=class_to_group,
                track=track,
                random_state=RANDOM_STATE,
            ).assign(clinical_track=track_name)
        )

    external_scores = pd.concat(external_score_parts, ignore_index=True)
    external_metrics = _summaries_by_configuration(
        external_scores,
        split="external",
        include_patient_average=False,
        seed_start=2026,
        notes="Development-locked model; external outcomes not used for selection",
    )
    for track_index, track_name in enumerate(CLINICAL_TRACKS):
        full = external_scores.loc[
            external_scores["clinical_track"].eq(track_name)
            & external_scores["configuration"].eq("ecology_plus_MLH1")
        ]
        for comparison_index, comparator_name in enumerate(
            ("ecology_only", "MLH1_only")
        ):
            comparator = external_scores.loc[
                external_scores["clinical_track"].eq(track_name)
                & external_scores["configuration"].eq(comparator_name)
            ]
            delta, replicates = _paired_delta_row(
                full,
                comparator,
                track_name,
                f"full_minus_{comparator_name}",
                "external",
                "attainable_development_locked_external",
                2026,
                source_runner=SOURCE_RUNNER,
            )
            attainable_delta_rows.append(delta)
            paired_replicate_parts.append(replicates)

    conditional_scores = pd.concat(conditional_score_parts, ignore_index=True)
    conditional_metrics, conditional_deltas, conditional_replicates = (
        _summarize_conditional_scores(conditional_scores)
    )
    paired_replicate_parts.append(conditional_replicates)
    paired_deltas = pd.concat(
        [pd.DataFrame(attainable_delta_rows), conditional_deltas], ignore_index=True
    )
    paired_replicates = pd.concat(paired_replicate_parts, ignore_index=True)
    _validate_bootstrap_replicates(paired_replicates, N_BOOTSTRAP)

    leave_one_scores = pd.concat(leave_one_score_parts, ignore_index=True)
    leave_one_metrics = _summarize_leave_one_out(leave_one_scores)

    clinical_score_parts: list[pd.DataFrame] = []
    for track_name in ("core_age_sex_side", "expanded_clinical6"):
        track = CLINICAL_TRACKS[track_name]
        _, scores = core.evaluate_clinical_baseline(
            train,
            external,
            metadata_train,
            metadata_external,
            outer_folds,
            list(track.covariates),
            RANDOM_STATE,
        )
        clinical_score_parts.append(
            scores.assign(
                clinical_track=track_name,
                configuration="clinical_only",
                selection_protocol="prespecified_clinical_logistic",
            )
        )
    clinical_scores = pd.concat(clinical_score_parts, ignore_index=True)
    clinical_metric_parts: list[pd.DataFrame] = []
    for index, ((track_name, split), part) in enumerate(
        clinical_scores.groupby(["clinical_track", "split"], sort=False)
    ):
        summary = summarize_scores(
            part,
            analysis=f"{track_name}_clinical_only_{split}",
            include_patient_average=split == "train_grouped_oof",
            random_state=2026,
            notes="Prespecified L2-logistic clinical-only comparator",
        )
        summary.insert(0, "clinical_track", track_name)
        summary.insert(1, "configuration", "clinical_only")
        summary.insert(2, "split", split)
        clinical_metric_parts.append(summary)
    clinical_metrics = pd.concat(clinical_metric_parts, ignore_index=True)

    all_attainable_scores = pd.concat(
        [nested_scores, external_scores], ignore_index=True, sort=False
    )
    clinical_track_metrics = attainable_metrics.loc[
        attainable_metrics["configuration"].eq("ecology_plus_MLH1")
    ].reset_index(drop=True)
    clinical_track_scores = nested_scores.loc[
        nested_scores["configuration"].eq("ecology_plus_MLH1")
    ].reset_index(drop=True)

    tables = {
        "clinical_track_nested_AUROC.csv": clinical_track_metrics,
        "clinical_track_nested_scores.csv": clinical_track_scores,
        "clinical_only_baseline_AUROC.csv": clinical_metrics,
        "clinical_only_baseline_scores.csv": clinical_scores,
        "attainable_ablation_AUROC.csv": attainable_metrics,
        "attainable_ablation_scores.csv": all_attainable_scores,
        "paired_ablation_delta_AUROC.csv": paired_deltas,
        "conditional_lineage_ablation_AUROC.csv": conditional_metrics,
        "conditional_lineage_ablation_scores.csv": conditional_scores,
        "exploratory_leave_one_feature_out_AUROC.csv": leave_one_metrics,
        "exploratory_leave_one_feature_out_scores.csv": leave_one_scores.loc[
            leave_one_scores["split"].isin(["train_grouped_oof", "external"])
        ].reset_index(drop=True),
        "external_ablation_AUROC.csv": external_metrics,
        "nested_fold_selections.csv": nested_selections,
        "nested_inner_candidate_AUROC.csv": nested_candidates,
        "paired_ablation_bootstrap_replicates.csv.gz": paired_replicates,
    }

    summary = build_analysis_summary(
        attainable_metrics, paired_deltas, external_metrics, clinical_metrics
    )
    after_hashes = snapshot_hashes(source_paths)
    manuscript_mtime_after = core.MANUSCRIPT_PATH.stat().st_mtime_ns
    if before_hashes != after_hashes or manuscript_mtime_before != manuscript_mtime_after:
        raise RuntimeError("a read-only source or manuscript changed during analysis")
    failure_count = sum(len(frame) for frame in failure_parts)
    manifest = build_manifest(
        output_dir=OUT,
        source_paths=source_paths,
        read_only_before=before_hashes,
        read_only_after=after_hashes,
        manuscript_mtime_before=manuscript_mtime_before,
        manuscript_mtime_after=manuscript_mtime_after,
        model_order=list(models),
        failure_count=failure_count,
    )
    manifest.update(
        {
            "cohort_contracts": {
                "GSE178341": {"specimens": 64, "patients": 62, "MSI": 35, "MSS": 29},
                "GSE132465": {"specimens": 23, "patients": 23, "MSI": 4, "MSS": 19},
            },
            "feature_contract": {
                "candidate_subtypes": len(candidates),
                "minimum_reference_cells": 50,
                "merged_subtypes": False,
                "screening_threshold": ">0.65 direction-corrected AUROC",
                "feature_prefix_range": "10 through min(24, n passing)",
                "MLH1_feature": core.MLH1_FEATURE,
            },
            "outer_fold_assignment": clinical_track_scores.loc[
                clinical_track_scores["clinical_track"].eq("expression_only"),
                ["sampleID", "patientID", "fold"],
            ].drop_duplicates().to_dict("records"),
            "development_locked_selections": {
                f"{track_name}/{configuration}": {
                    "feature_set": selection.feature_set,
                    "subtype_features": list(selection.subtype_features),
                    "model": selection.model_name,
                    "grouped_oof_auc": selection.inner_auc,
                    "n_passing_screen": selection.n_passing_screen,
                }
                for (track_name, configuration), selection in development_selections.items()
            },
            "required_csv_outputs": list(REQUIRED_CSV_OUTPUTS),
        }
    )
    write_outputs(OUT, tables, manifest, summary)
    print(f"Output: {OUT}", flush=True)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
