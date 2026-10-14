#!/usr/bin/env python3
"""Evaluate unmerged scRNA subtypes after removing reference classes with <50 cells."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from run_unified66_scrna_pipeline import (
    RANDOM_STATE,
    ClassContract,
    _contract_table,
    _external_exploratory_ranking,
    _feature_set_table,
    _identity_mask,
    _json_scalar,
    attach_mlh1_features,
    build_feature_sets,
    load_gse132465_cells,
    load_gse178341_cells,
    make_models,
    rank_train_only,
    run_model_grid,
    screen_features,
    select_auc_features,
)


FINAL = Path(__file__).resolve().parents[1]
ROOT = FINAL.parent
OUT = (
    FINAL
    / "data"
    / "scRNA"
    / "result"
    / "diagnostics"
    / "unmerged_raw_subtypes_min50_20260908"
)
SPLIT_TNKILC = (
    "cTNI08",
    "cTNI09",
    "cTNI14",
    "cTNI16",
    "cTNI18",
    "cTNI19",
    "cTNI20",
    "cTNI21",
)


def load_unmerged_class_contract(
    definitions_path: str | Path,
    counts_path: str | Path,
    min_cells: int = 50,
) -> ClassContract:
    definitions = pd.read_csv(definitions_path, dtype=str)
    counts = pd.read_csv(counts_path)
    required_definitions = {"feature", "feature_type", "group_class", "subclass"}
    required_counts = {"CellType", "n_cells"}
    if not required_definitions.issubset(definitions.columns):
        missing = sorted(required_definitions.difference(definitions.columns))
        raise ValueError(f"feature definitions missing columns: {missing}")
    if not required_counts.issubset(counts.columns):
        missing = sorted(required_counts.difference(counts.columns))
        raise ValueError(f"cell counts missing columns: {missing}")

    raw = definitions.loc[definitions["feature_type"].eq("raw_subtype")].copy()
    raw["feature"] = raw["feature"].astype(str).str.strip()
    raw["subclass"] = raw["subclass"].astype(str).str.strip()
    raw["group_class"] = raw["group_class"].astype(str).str.strip()
    if raw["feature"].duplicated().any() or raw["subclass"].duplicated().any():
        raise ValueError("duplicate raw-subtype definitions")
    if not raw["feature"].eq(raw["subclass"]).all():
        mismatched = raw.loc[~raw["feature"].eq(raw["subclass"]), ["feature", "subclass"]]
        raise ValueError(f"raw feature/subclass names differ: {mismatched.to_dict('records')}")

    subtype_to_group = dict(zip(raw["feature"], raw["group_class"]))
    modules = definitions.loc[definitions["feature_type"].eq("module")].copy()
    for module in modules.itertuples(index=False):
        group_class = str(module.group_class).strip()
        members = [
            member.strip()
            for member in str(module.member_subclasses).split(";")
            if member.strip()
        ]
        for member in members:
            previous = subtype_to_group.get(member)
            if previous is not None and previous != group_class:
                raise ValueError(
                    f"conflicting group definitions for {member}: {previous}, {group_class}"
                )
            subtype_to_group[member] = group_class

    counts["CellType"] = counts["CellType"].astype(str).str.strip()
    counts["n_cells"] = pd.to_numeric(counts["n_cells"], errors="raise").astype(int)
    if counts["CellType"].duplicated().any():
        raise ValueError("duplicate cell-count definitions")
    retained = counts.loc[counts["n_cells"].ge(int(min_cells))].copy()
    class_names = tuple(retained["CellType"].tolist())
    undefined = sorted(set(class_names).difference(subtype_to_group))
    if undefined:
        raise ValueError(f"retained raw subtypes missing definitions: {undefined}")

    return ClassContract(
        class_names=class_names,
        class_to_group={name: subtype_to_group[name] for name in class_names},
        subtype_to_class={name: name for name in subtype_to_group},
        cell_counts=dict(zip(retained["CellType"], retained["n_cells"])),
    )


def build_raw_group_fraction_matrix(
    cells: pd.DataFrame,
    sample_labels: pd.DataFrame,
    contract: ClassContract,
) -> pd.DataFrame:
    required_cells = {"sampleID", "raw_subtype", "major_class"}
    required_labels = {"sampleID", "MMR"}
    if not required_cells.issubset(cells.columns):
        raise ValueError(
            f"cell table missing columns: {sorted(required_cells.difference(cells.columns))}"
        )
    if not required_labels.issubset(sample_labels.columns):
        raise ValueError(
            f"sample-label table missing columns: {sorted(required_labels.difference(sample_labels.columns))}"
        )

    labels = sample_labels.loc[:, ["sampleID", "MMR"]].copy()
    labels["sampleID"] = labels["sampleID"].astype(str)
    labels["MMR"] = labels["MMR"].astype(str)
    conflicts = labels.groupby("sampleID", sort=False)["MMR"].nunique(dropna=False)
    if conflicts.gt(1).any():
        raise ValueError(f"conflicting labels for samples: {sorted(conflicts[conflicts.gt(1)].index)}")
    labels = labels.drop_duplicates("sampleID", keep="first").set_index("sampleID")

    working = cells.loc[:, ["sampleID", "raw_subtype", "major_class"]].copy()
    working["sampleID"] = working["sampleID"].astype(str)
    working["raw_subtype"] = working["raw_subtype"].astype("string")
    working["major_class"] = working["major_class"].astype("string")
    working = working.loc[working["raw_subtype"].notna() & working["major_class"].notna()].copy()
    unknown = sorted(
        set(working["raw_subtype"].astype(str)).difference(contract.subtype_to_class)
    )
    if unknown:
        raise ValueError(f"unknown raw subtypes: {unknown}")
    unknown_samples = sorted(set(working["sampleID"]).difference(labels.index))
    if unknown_samples:
        raise ValueError(f"cells lack sample labels: {unknown_samples}")

    retained = working.loc[working["raw_subtype"].isin(contract.class_names)].copy()
    retained["expected_group"] = retained["raw_subtype"].map(contract.class_to_group)
    mismatch = retained["expected_group"].ne(retained["major_class"])
    if mismatch.any():
        examples = retained.loc[
            mismatch, ["raw_subtype", "major_class", "expected_group"]
        ].drop_duplicates()
        raise ValueError(f"raw subtype/major-class mismatch: {examples.to_dict('records')}")

    matrix = pd.DataFrame(0.0, index=labels.index, columns=contract.class_names)
    if not retained.empty:
        denominators = (
            working.groupby(["sampleID", "major_class"], observed=True)
            .size()
            .rename("denominator")
            .reset_index()
        )
        numerators = (
            retained.groupby(["sampleID", "major_class", "raw_subtype"], observed=True)
            .size()
            .rename("numerator")
            .reset_index()
        )
        fractions = numerators.merge(
            denominators,
            on=["sampleID", "major_class"],
            how="left",
            validate="many_to_one",
        )
        fractions["fraction"] = fractions["numerator"] / fractions["denominator"]
        pivot = fractions.pivot(index="sampleID", columns="raw_subtype", values="fraction")
        matrix.loc[pivot.index, pivot.columns] = pivot
    matrix = matrix.fillna(0.0).astype(float)
    matrix.index.name = "sampleID"
    matrix.insert(0, "MMR", labels.loc[matrix.index, "MMR"].to_numpy())
    return matrix


def compute_external_univariate_auc(
    train_stats: pd.DataFrame,
    external: pd.DataFrame,
) -> pd.DataFrame:
    y = external["MMR"].astype(str).eq("MSI").to_numpy(dtype=int)
    if len(np.unique(y)) != 2:
        raise ValueError("external AUROC requires both MSI and MSS samples")

    rows = []
    for row in train_stats.itertuples(index=False):
        feature = str(row.feature)
        values = pd.to_numeric(external[feature], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"external feature contains non-finite values: {feature}")
        external_auc = float(roc_auc_score(y, values))
        train_direction = "MSI_up" if float(row.auc_msi_high) >= 0.5 else "MSI_down"
        same_direction_auc = (
            external_auc if train_direction == "MSI_up" else 1.0 - external_auc
        )
        rows.append(
            {
                "feature": feature,
                "group_class": getattr(row, "group_class", ""),
                "train_auc_msi_high": float(row.auc_msi_high),
                "train_auc_best_direction": float(row.auc_best_direction),
                "train_direction_by_auc": train_direction,
                "external_auc_msi_high": external_auc,
                "external_auc_same_direction": same_direction_auc,
                "external_auc_best_direction": max(external_auc, 1.0 - external_auc),
                "direction_maintained": bool(same_direction_auc > 0.5),
            }
        )
    return pd.DataFrame(rows)


def tosica_migration_coverage(migration_path: str | Path) -> pd.DataFrame:
    columns = ["Class", "major_prediction", "final_migrated_subtype"]
    migration = pd.read_csv(migration_path, usecols=columns)
    tumor = migration.loc[migration["Class"].astype("string").str.lower().eq("tumor")].copy()
    grouped = tumor.groupby("major_prediction", dropna=False, observed=True)
    rows = []
    for major_class, group in grouped:
        migrated = group["final_migrated_subtype"].notna()
        rows.append(
            {
                "major_class": major_class,
                "n_tumor_cells": int(len(group)),
                "n_cells_with_subtype_prediction": int(migrated.sum()),
                "subtype_prediction_coverage": float(migrated.mean()),
                "n_predicted_subtypes": int(group.loc[migrated, "final_migrated_subtype"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values("major_class", kind="mergesort").reset_index(drop=True)


def main() -> None:
    definitions_path = FINAL / "data" / "scRNA" / "reference" / "hybrid_feature_definitions.csv"
    counts_path = (
        FINAL
        / "data"
        / "bulk"
        / "prepare"
        / "GSE178341_DECODE_reference"
        / "noncompressed_min50"
        / "reference"
        / "gse178341_decode_reference_tumor_full_protein_coding_celltype_counts.csv"
    )
    train_processed_path = FINAL / "data" / "scRNA" / "result" / "process" / "GSE178341_processed.csv"
    train_mlh1_path = FINAL / "data" / "scRNA" / "input" / "gse178341_18_plus_mlh1_features.csv"
    external_processed_path = FINAL / "data" / "scRNA" / "result" / "process" / "GSE132465_processed.csv"
    external_migration_path = (
        FINAL / "data" / "scRNA" / "result" / "process" / "GSE132465_tosica_migration_cells.csv"
    )

    contract = load_unmerged_class_contract(definitions_path, counts_path, min_cells=50)
    train_cells, train_labels = load_gse178341_cells(ROOT / "GSE178341.h5ad")
    external_cells, external_labels = load_gse132465_cells(
        external_migration_path, external_processed_path
    )
    train_matrix = build_raw_group_fraction_matrix(train_cells, train_labels, contract)
    external_matrix = build_raw_group_fraction_matrix(external_cells, external_labels, contract)
    train_matrix = attach_mlh1_features(train_matrix, [train_mlh1_path, train_processed_path])
    external_matrix = attach_mlh1_features(external_matrix, [external_processed_path])

    expected_train = {"MSI": 35, "MSS": 29}
    expected_external = {"MSS": 19, "MSI": 4}
    if len(train_matrix) != 64 or train_matrix["MMR"].value_counts().to_dict() != expected_train:
        raise ValueError(f"unexpected GSE178341 cohort: {train_matrix['MMR'].value_counts().to_dict()}")
    if len(external_matrix) != 23 or external_matrix["MMR"].value_counts().to_dict() != expected_external:
        raise ValueError(f"unexpected GSE132465 cohort: {external_matrix['MMR'].value_counts().to_dict()}")

    stats = screen_features(
        train_matrix.reset_index(),
        list(contract.class_names),
        class_to_group=contract.class_to_group,
    )
    selected_features = select_auc_features(stats, threshold=0.65)
    all_external_auc = compute_external_univariate_auc(stats, external_matrix.reset_index())
    external_auc = all_external_auc.loc[
        all_external_auc["feature"].isin(selected_features)
    ].copy()
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
            metrics.loc[_identity_mask(metrics, primary)].assign(selection="train_selected_primary"),
            metrics.loc[_identity_mask(metrics, external_best)].assign(
                selection="external_best_exploratory"
            ),
        ],
        ignore_index=True,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    _contract_table(contract).to_csv(OUT / "class_contract_71_unmerged.csv", index=False)
    train.insert(0, "dataset", "GSE178341")
    external.insert(0, "dataset", "GSE132465")
    train.to_csv(OUT / "GSE178341_unmerged71_features.csv", index=False)
    external.to_csv(OUT / "GSE132465_unmerged71_features.csv", index=False)
    stats.to_csv(OUT / "GSE178341_unmerged71_univariate_auc.csv", index=False)
    selected_stats = stats.loc[stats["feature"].isin(selected_features)].copy()
    selected_stats.to_csv(OUT / "GSE178341_unmerged71_auc_gt065.csv", index=False)
    all_external_auc.to_csv(OUT / "GSE132465_unmerged71_univariate_auc.csv", index=False)
    external_auc.to_csv(OUT / "GSE132465_selected_feature_univariate_auc.csv", index=False)
    all_external_auc.loc[all_external_auc["feature"].isin(SPLIT_TNKILC)].to_csv(
        OUT / "split_tnk_subtype_validation_auc.csv", index=False
    )
    tosica_migration_coverage(external_migration_path).to_csv(
        OUT / "GSE132465_tosica_migration_coverage.csv", index=False
    )
    _feature_set_table(feature_sets).to_csv(OUT / "feature_sets.csv", index=False)
    metrics.to_csv(OUT / "model_grid_metrics.csv", index=False)
    predictions.to_csv(OUT / "model_grid_sample_predictions.csv", index=False)
    importance.to_csv(OUT / "model_grid_feature_importance_or_coef.csv", index=False)
    failures.to_csv(OUT / "model_grid_failures.csv", index=False)
    ranked.to_csv(OUT / "model_grid_ranked_train_only.csv", index=False)
    external_ranked.to_csv(OUT / "model_grid_ranked_external_exploratory.csv", index=False)
    selected_metrics.to_csv(OUT / "selected_model_metrics.csv", index=False)

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "output_directory": str(OUT),
        "random_state": RANDOM_STATE,
        "class_contract": {
            "n_classes": len(contract.class_names),
            "merge_subtypes": False,
            "minimum_reference_cells_inclusive": 50,
            "fraction_denominator": "all subtype-labelled cells within each major class",
        },
        "screening": {
            "dataset": "GSE178341",
            "threshold_rule": "auc_best_direction > 0.65",
            "feature_selection_scope": "full development cohort before fixed-feature OOF",
            "n_selected": len(selected_features),
            "selected_features_ranked": selected_features,
        },
        "cohorts": {"GSE178341": expected_train, "GSE132465": expected_external},
        "tosica": {
            "migration_file": str(external_migration_path),
            "reused_existing_predictions": True,
            "subtype_models": ["TNKILC", "Myeloid", "Strom", "Plasma"],
            "major_only_without_subtype_model": ["B", "Mast"],
        },
        "model_protocol": {
            "always_included_feature": "mean_MLH1_epithelial",
            "normalization": "MinMaxScaler fit on each fold training split; final scaler fit on all GSE178341",
            "primary_selection": "training OOF AUC, OOF balanced accuracy, fewer subtype features, stable model order",
            "external_selection_status": "exploratory only; not used for primary selection",
        },
        "train_selected_primary": {
            key: _json_scalar(primary[key]) for key in ["feature_set", "model", "n_subtype_features"]
        },
        "external_best_exploratory": {
            key: _json_scalar(external_best[key])
            for key in ["feature_set", "model", "n_subtype_features", "auc", "balanced_accuracy"]
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
    print(f"unmerged_classes={len(contract.class_names)}")
    print(f"auc_gt_0.65_features={len(selected_features)} {selected_features}")
    print(
        f"train_selected_primary={primary['feature_set']} {primary['model']} "
        f"oof_auc={primary['train_oof_auc']:.6f} "
        f"oof_ba={primary['train_oof_balanced_accuracy']:.6f} "
        f"external_auc={primary['external_auc']:.6f} "
        f"external_ba={primary['external_balanced_accuracy']:.6f}"
    )
    print(
        f"external_best_exploratory={external_best['feature_set']} {external_best['model']} "
        f"auc={external_best['auc']:.6f} ba={external_best['balanced_accuracy']:.6f}"
    )


if __name__ == "__main__":
    main()
