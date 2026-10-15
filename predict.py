"""MSIClassifier: expression matrix to features to frozen classifier scores."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "workflows/expression"))
from inputs import read_bulk, map_probes, bulk_scales, read_cells, log_expression, require_counts


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="modality", required=True)
    for name in ["bulk", "scrna", "spatial"]:
        s = sub.add_parser(name)
        s.add_argument("--expression", type=Path, required=True)
        s.add_argument("--output-dir", type=Path, required=True, help="New output directory for this run")
        s.add_argument("--check-only", action="store_true", help="Inspect input only; no model execution")
        s.add_argument("--resource-dir", type=Path, default=ROOT / "resources/raw_reference")
        if name == "bulk":
            s.add_argument("--scale", required=True, choices=["linear", "log2", "log2p1"])
            s.add_argument("--dataset", required=True)
            s.add_argument("--probe-map", type=Path, help="CSV with probe_id,gene_symbol; median probes in supplied scale")
            s.add_argument("--workers", type=int, default=4)
            s.add_argument("--mmr4-lock", type=Path, help="Matching existing-cohort evaluation lock only")
        else:
            s.add_argument("--format", default="auto", choices=["auto", "h5ad", "10x-h5", "10x-mtx", "loom", "text"])
            s.add_argument("--transpose", action="store_true", help="Transpose text gene-by-cell/spot matrix")
            s.add_argument("--layer", help="Explicit AnnData expression layer; default X")
            s.add_argument("--gene-symbol-column", help="var column to use instead of var_names")
            s.add_argument("--device", default="cpu", choices=["cpu", "cuda:0"])
            if name == "scrna":
                s.add_argument("--scale", required=True, choices=["counts", "log1p-cp10k"])
                s.add_argument("--dataset", required=True)
                s.add_argument("--metadata", type=Path, help="CSV indexed by cell_id with Sample and Class")
                s.add_argument("--sample-column", default="Sample")
                s.add_argument("--tissue-column", default="Class")
                s.add_argument("--tumor-value", default="Tumor")
                s.add_argument("--batch-size", type=int, default=1024)
            else:
                s.add_argument("--sample-id", required=True, help="One section per invocation")
                s.add_argument("--positions", type=Path, help="Optional CSV indexed by spot_id, columns x,y")
    return p


def main():
    args = parser().parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory must be new or empty; preserve prior results")
    out.mkdir(parents=True, exist_ok=True)
    report = {"modality": args.modality, "input": str(args.expression.resolve()),
              "model_executed": False, "seed": 2026}
    if args.modality == "bulk":
        if args.workers < 1:
            raise ValueError("--workers must be positive")
        genes = read_bulk(args.expression)
        if args.probe_map:
            genes = map_probes(genes, pd.read_csv(args.probe_map))
        linear, sampling, logs = bulk_scales(genes, args.scale)
        report.update(n_samples=genes.shape[1], n_genes=genes.shape[0], expression_scale=args.scale,
                      dataset=args.dataset, probe_mapping=bool(args.probe_map))
        data = (linear, sampling, logs)
    else:
        a = read_cells(args.expression, args.format, args.transpose, args.layer)
        if args.gene_symbol_column:
            if args.gene_symbol_column not in a.var or a.var[args.gene_symbol_column].isna().any():
                raise ValueError("Missing gene-symbol var column")
            a.var_names = a.var[args.gene_symbol_column].astype(str)
            if not a.var_names.is_unique:
                raise ValueError("Duplicate gene symbols; resolve before normalization")
        if args.modality == "scrna":
            if args.metadata:
                meta = pd.read_csv(args.metadata, index_col=0, dtype=str)
                if not meta.index.is_unique or not a.obs_names.isin(meta.index).all():
                    raise ValueError("Metadata must uniquely cover every cell_id")
                for col in meta:
                    a.obs[col] = meta.reindex(a.obs_names)[col].to_numpy()
            required = [args.sample_column, args.tissue_column]
            if any(c not in a.obs for c in required) or a.obs[required].isna().any().any():
                raise ValueError("Missing sample/tissue metadata: " + ", ".join(required))
            a.obs["sampleID"] = a.obs[args.sample_column].astype(str)
            a.obs["is_tumor"] = a.obs[args.tissue_column].astype(str).eq(args.tumor_value)
            a = log_expression(a, args.scale)
            report.update(n_cells=a.n_obs, n_genes=a.n_vars, n_samples=a.obs.sampleID.nunique(),
                          expression_scale=args.scale, sample_QC_min_total_cells=200)
        else:
            require_counts(a)
            if args.positions:
                xy = pd.read_csv(args.positions, index_col=0)
                if not xy.index.is_unique or not a.obs_names.isin(xy.index).all() or any(c not in xy for c in ["x", "y"]):
                    raise ValueError("Position table must uniquely cover every spot_id with x,y")
                coords = xy.reindex(a.obs_names)[["x", "y"]].to_numpy(dtype=float)
                if not np.isfinite(coords).all():
                    raise ValueError("Non-finite spatial coordinates")
                a.obsm["spatial"] = coords
            report.update(n_spots=a.n_obs, n_genes=a.n_vars, sample_id=args.sample_id,
                          expression_scale="counts", spatial_coordinates_available="spatial" in a.obsm)
        data = a
    (out / "input_check.json").write_text(json.dumps(report, indent=2))
    if args.check_only:
        print(json.dumps(report))
        return
    from runners import run_bulk, run_scrna, run_spatial
    {"bulk": run_bulk, "scrna": run_scrna, "spatial": run_spatial}[args.modality](args, data, out)


if __name__ == "__main__":
    main()
