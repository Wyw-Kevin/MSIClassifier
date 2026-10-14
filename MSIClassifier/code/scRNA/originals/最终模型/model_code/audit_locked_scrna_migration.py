"""Audit frozen query annotation without changing historical predictions."""

import hashlib
import json
import random
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse

FINAL = Path(__file__).resolve().parents[1]
ROOT = FINAL.parent
OUT = FINAL / "data/scRNA/result/diagnostics/locked_workflow_migration_audit"
sys.path.insert(0, str(FINAL / "scRNA_model"))
from scRNA_msi import preprocess as pp
import run_scrna_metadata_adjusted_pipeline as core

SEED = 2026


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    contract = pd.read_csv(core.CLASS_CONTRACT_PATH)
    parent = dict(zip(contract.class_name, contract.group_class))
    summary, checks, impacts, models = [], [], [], []
    protocol = json.loads((FINAL / "data/scRNA/result/diagnostics/GSE200997_locked_shared18/frozen_protocol.json").read_text())
    for dataset, process in [
        ("GSE132465", FINAL / "data/scRNA/result/process"),
        ("GSE200997", FINAL / "data/scRNA/result/diagnostics/GSE200997_locked_shared18/process"),
    ]:
        manifest = json.loads((process / f"{dataset}_migration_manifest.json").read_text())
        cells = pd.read_csv(process / f"{dataset}_tosica_migration_cells.csv")
        a = ad.read_h5ad(manifest["h5ad"], backed="r")
        assert cells.cell_id.astype(str).tolist() == a.obs_names.astype(str).tolist()
        tumor = cells.Class.str.lower().eq("tumor")
        annotated = cells.final_migrated_subtype.notna()
        low_major = cells.major_probability.lt(0.5)
        low_subtype = cells.subtype_probability.lt(0.5) & annotated
        summary.append(dict(dataset=dataset, n_cells=len(cells), tumor_cells=int(tumor.sum()),
            tumor_samples=int(cells.loc[tumor, "Sample"].nunique()),
            tumor_major_below_05=int((tumor & low_major).sum()),
            tumor_subtyped=int((tumor & annotated).sum()),
            tumor_subtype_below_05=int((tumor & low_subtype).sum()),
            tumor_shared18_below_05=int((tumor & cells.final_migrated_subtype.isin(core.CURRENT_SHARED18) & (low_major | low_subtype)).sum()),
            confidence_cutoff_status="0.5 illustrative only; not a selected production cutoff"))
        # Quantify potential denominator and rejection effects without assigning new labels.
        for sid, sub in cells.loc[tumor].groupby("Sample"):
            for feature in core.CURRENT_SHARED18:
                in_parent = sub.major_prediction.eq(parent[feature])
                denominator = int(in_parent.sum())
                selected = sub.final_migrated_subtype.eq(feature)
                rejected = selected & (sub.major_probability.lt(0.5) | sub.subtype_probability.lt(0.5))
                impacts.append(dict(dataset=dataset, sampleID=sid, feature=feature,
                    parent=parent[feature], parent_count=denominator,
                    feature_count=int(selected.sum()), below_05_count=int(rejected.sum()),
                    original_fraction=int(selected.sum()) / denominator if denominator else None,
                    illustrative_fraction_loss=int(rejected.sum()) / denominator if denominator else None))
        for info in manifest["model_infos"]:
            directory = Path(info["model_dir"])
            weight = directory / info["weight"]
            sha = digest(weight)
            expected_sha = protocol["migration_weight_sha256"].get(str(weight))
            assert expected_sha == sha, (dataset, weight, "changed checkpoint")
            model_manifest = directory / "model_manifest.json"
            train_meta = json.loads(model_manifest.read_text()) if model_manifest.exists() else {}
            models.append(dict(dataset=dataset, model=info["model"], weight=str(weight),
                sha256=sha, historical_training_seed=train_meta.get("seed", "unrecorded"),
                n_genes=info["n_genes"], missing_genes=info["missing_genes"],
                missing_fraction=info["missing_genes"] / info["n_genes"]))
            candidates = np.arange(len(cells)) if info["model"] == "Major" else np.flatnonzero(cells.major_prediction.eq(info["model"]))
            # Deterministic spread across stored row order; no random cell selection.
            idx = candidates[np.linspace(0, len(candidates) - 1, min(24, len(candidates)), dtype=int)]
            outputs = []
            for _ in range(2):
                seed_all()
                model, labels, genes, _ = pp.load_tosica_model(directory, info["weight"], torch.device("cpu"))
                batch = pp.GenePlan(a.var_names, genes).load_batch(a, idx)
                with torch.no_grad():
                    probabilities = torch.softmax(pp.logits_only(model, torch.from_numpy(batch)), dim=1)
                probability, predicted = probabilities.max(dim=1)
                outputs.append((np.array(labels)[predicted.numpy()], probability.numpy()))
                del model
            column = "major_prediction" if info["model"] == "Major" else "subtype_prediction"
            pcolumn = "major_probability" if info["model"] == "Major" else "subtype_probability"
            checks.append(dict(dataset=dataset, model=info["model"], n_cells=len(idx),
                repeated_labels_equal=bool(np.array_equal(outputs[0][0], outputs[1][0])),
                repeated_probability_max_difference=float(np.max(np.abs(outputs[0][1] - outputs[1][1]))),
                matches_historical_labels=int(np.sum(outputs[0][0] == cells.iloc[idx][column].to_numpy())),
                historical_probability_max_difference=float(np.max(np.abs(outputs[0][1] - cells.iloc[idx][pcolumn].to_numpy())))))
            print(dataset, info["model"], checks[-1], flush=True)
        # Inspect stored input scale; this does not establish historical normalization provenance.
        rows = np.linspace(0, a.n_obs - 1, min(128, a.n_obs), dtype=int)
        x = a[rows, :].X
        values = x.data if sparse.issparse(x) else np.asarray(x).ravel()
        summary[-1].update(input_min=float(values.min()), input_max=float(values.max()),
            sampled_noninteger_fraction=float(np.mean(np.abs(values - np.round(values)) > 1e-6)),
            layers=list(a.layers.keys()), raw_available=a.raw is not None)
        a.file.close()
    pd.DataFrame(summary).to_csv(OUT / "cohort_audit.csv", index=False)
    pd.DataFrame(models).to_csv(OUT / "checkpoint_audit.csv", index=False)
    pd.DataFrame(checks).to_csv(OUT / "frozen_inference_replay.csv", index=False)
    pd.DataFrame(impacts).to_csv(OUT / "shared18_confidence_and_denominators.csv", index=False)
    report = dict(seed=SEED, contract_n=len(contract), contract_min_cells=int(contract.reference_n_cells.min()),
        cohort_audit=summary, inference_checks=checks,
        scope="Read-only historical inputs; 24 cells/model/cohort replayed twice on CPU. No full migration or retraining. No new AUROC claimed.")
    (OUT / "audit.json").write_text(json.dumps(report, indent=2))
    assert all(row["repeated_labels_equal"] and row["repeated_probability_max_difference"] == 0 for row in checks)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
