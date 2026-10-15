"""Modality execution routes; existing frozen classifiers are never refitted."""
from pathlib import Path
import hashlib
import importlib.util
import importlib.metadata
import json
import random
import sys
import numpy as np
import pandas as pd
from inputs import PANEL12, PANEL18, parent

ROOT = Path(__file__).resolve().parents[2]


def source_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest(out, values):
    versions = {}
    for name in ["numpy", "pandas", "scikit-learn", "scanpy", "infercnvpy", "torch", "cell2location", "scvi-tools"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    payload = {"seed": 2026, "versions": versions, "label_use": "No MSI labels used for preprocessing, adaptation or prediction", **values}
    (out / "run_manifest.json").write_text(json.dumps(payload, indent=2))


def require_resources(directory, prefixes):
    directory = Path(directory)
    path = directory / "manifest.json"
    if not path.exists():
        raise FileNotFoundError("Install the separate raw-reference archive at " + str(directory) + "; see docs/EXPRESSION_WORKFLOW.md")
    records = json.loads(path.read_text())["files"]
    used = {k: v for k, v in records.items() if any(k.startswith(p) for p in prefixes)}
    if not used:
        raise ValueError("Required reference resources not present in manifest")
    for name, record in used.items():
        file = directory / name
        if not file.exists() or hashlib.sha256(file.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Frozen reference missing or changed: " + name)
    return used


def score_features(features, out, modality, lock=None):
    features.to_csv(out / "features.csv", index=False, float_format="%.17g")
    import subprocess
    if modality == "bulk":
        command = [sys.executable, str(ROOT / "workflows/bulk/predict_bulk.py"),
                   "--feature-table", str(out / "features.csv"), "--output", str(out / "scores.csv")]
        if lock:
            command += ["--mmr4-lock", str(lock.resolve())]
    else:
        command = [sys.executable, str(ROOT / "workflows/scRNA/predict_scrna.py"),
                   "--feature-table", str(out / "features.csv"), "--output", str(out / "scores.csv")]
    subprocess.run(command, check=True)


def run_bulk(args, data, out):
    """Fixed raw-count reference, independent target-cohort BayesPrism update."""
    linear, sampling, logs = data
    bp = ROOT / "workflows/bayesprism"
    sys.path.insert(0, str(bp / "vendor"))
    sys.path.insert(0, str(bp))
    import runtime
    from pybayesprism import prism
    ref = pd.read_pickle(bp / "resources/reference/raw_count_profiles.pkl")
    mapping = pd.read_csv(bp / "resources/reference/mapping.csv", index_col=0)
    overlap = ref.columns.intersection(sampling.index)
    if len(overlap) < 1000:
        raise ValueError(f"Only {len(overlap)} reference genes overlap; at least 1000 required by the port")
    if (sampling.loc[overlap].sum(axis=0) <= 0).any():
        raise ValueError("A sample has zero rounded library across shared reference genes")
    random.seed(2026); np.random.seed(2026); runtime.install(out)
    linear.T.to_pickle(out / "linear_expression.pkl")
    sampling.T.to_pickle(out / "sampling_expression.pkl")
    p = prism.Prism.new(ref, "GEP", ref.index.tolist(), ref.index.tolist(), None, sampling.T)
    r = p.run(n_cores=args.workers, update_gibbs=True,
              gibbs_control={"seed": 2026, "chain.length": 1000, "burn.in": 500, "thinning": 2, "alpha": 1})
    abundance = r.posterior_theta_f.theta
    if not abundance.index.equals(sampling.columns) or not np.isfinite(abundance.to_numpy()).all() or (abundance < 0).any().any() or not np.allclose(abundance.sum(axis=1), 1):
        raise ValueError("Invalid final BayesPrism abundance output")
    abundance.to_csv(out / "final_fractions.csv", index_label="sampleID")
    features = pd.DataFrame(index=abundance.index)
    for feature in PANEL12:
        denominator = abundance[mapping.index[mapping.parent.eq(parent(feature))]].sum(axis=1)
        if (denominator <= 0).any():
            raise ValueError("Missing parent denominator for " + feature)
        features[feature] = abundance[feature] / denominator
    features.index.name = "sampleID"
    features = features.reset_index()
    features.insert(0, "dataset", args.dataset)
    if args.mmr4_lock:
        for gene in ["MLH1", "MSH2", "MSH6", "PMS2"]:
            if gene not in logs.index:
                raise ValueError("MMR4 gene absent: " + gene)
            features[gene + "_raw"] = logs.loc[gene, features.sampleID].to_numpy()
    score_features(features, out, "bulk", args.mmr4_lock)
    manifest(out, {"modality": "bulk", "n_samples": len(features), "reference_genes_overlap": len(overlap),
                   "scale": args.scale, "input_rounding": "np.rint once after explicit inverse transform",
                   "output_abundance": "posterior_theta_f.theta", "key": None, "update_gibbs": True,
                   "target_cohort_adaptation": True, "sampling": {"chain": 1000, "burn": 500, "thin": 2, "alpha": 1, "sigma": 2},
                   "fusion": "fixed0.5 with matching existing-cohort lock" if args.mmr4_lock else "not computed",
                   "interpretation": "New target-cohort unsupervised adaptation; microarray intensities are count proxies. No new validation performance claimed."})


def run_scrna(args, a, out):
    from scrna import run
    run(args, a, out)


def run_spatial(args, a, out):
    from spatial import run
    run(args, a, out)
