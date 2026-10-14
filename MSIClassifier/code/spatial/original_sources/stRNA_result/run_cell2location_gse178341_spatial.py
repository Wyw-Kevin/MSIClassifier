#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run GSE178341 -> Visium deconvolution with cell2location.

Outputs are written as one folder per spatial sample under stRNA_result/.
Small classes use GSE178341 cl295v11SubShort labels; major classes are summed
from majorCluster.
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from cell2location.models import Cell2location, RegressionModel
from cell2location.utils.filtering import filter_genes


ROOT = Path("/home/wyw/rstudio/MSI")
SPATIAL_ROOT = ROOT / "空间转录组"
OUTPUT_ROOT = ROOT / "stRNA_result"
GSE178341_H5AD = ROOT / "GSE178341.h5ad"
COMMON_GENES = ROOT / "bulk_data/standardized_expression_common_genes.txt"

SMALL_KEY = "cl295v11SubShort"
MAJOR_KEY = "majorCluster"
SAMPLE_KEY = "sample"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spatial-root", type=Path, default=SPATIAL_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--reference-h5ad", type=Path, default=GSE178341_H5AD)
    parser.add_argument("--common-genes", type=Path, default=COMMON_GENES)
    parser.add_argument("--samples", default="", help="Comma-separated sample folder names/substrings to run.")
    parser.add_argument("--max-cells-per-small", type=int, default=1200)
    parser.add_argument("--min-cells-per-small", type=int, default=30)
    parser.add_argument("--signature-method", choices=["regression", "mean"], default="regression")
    parser.add_argument("--reference-epochs", type=int, default=180)
    parser.add_argument("--spatial-epochs", type=int, default=5000)
    parser.add_argument("--posterior-samples", type=int, default=300)
    parser.add_argument("--reference-batch-size", type=int, default=2500)
    parser.add_argument("--spatial-batch-size", type=int, default=1024)
    parser.add_argument("--n-cells-per-location", type=float, default=8.0)
    parser.add_argument("--detection-alpha", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--force-reference", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_common_genes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {x.strip() for x in path.read_text().splitlines() if x.strip()}


def list_spatial_samples(root: Path, sample_filter: str = "") -> list[Path]:
    samples = [p for p in sorted(root.iterdir()) if p.is_dir() and list(p.glob("*filtered_feature_bc_matrix.h5"))]
    if not sample_filter:
        return samples
    wanted = [x.strip() for x in sample_filter.split(",") if x.strip()]
    return [p for p in samples if any(w in p.name for w in wanted)]


def get_spatial_gene_symbols(sample_dir: Path) -> set[str]:
    h5_path = next(sample_dir.glob("*filtered_feature_bc_matrix.h5"))
    adata = sc.read_10x_h5(h5_path)
    return set(map(str, adata.var_names))


def collapse_duplicate_genes(adata: ad.AnnData) -> ad.AnnData:
    symbols = pd.Index(map(str, adata.var_names))
    if not symbols.has_duplicates:
        adata.var_names = symbols
        return adata

    codes, unique = pd.factorize(symbols, sort=False)
    mapper = sp.csr_matrix(
        (np.ones(len(codes), dtype=np.float32), (np.arange(len(codes)), codes)),
        shape=(len(codes), len(unique)),
    )
    x = adata.X
    if not sp.issparse(x):
        x = sp.csr_matrix(x)
    x_new = (x.tocsr() @ mapper).astype(np.float32)
    out = ad.AnnData(
        X=x_new,
        obs=adata.obs.copy(),
        var=pd.DataFrame(index=pd.Index(unique, name=adata.var_names.name)),
    )
    for key, value in adata.obsm.items():
        out.obsm[key] = value.copy() if hasattr(value, "copy") else value
    out.uns = dict(adata.uns)
    return out


def sample_reference_cells(obs: pd.DataFrame, max_cells: int, min_cells: int, seed: int) -> tuple[np.ndarray, pd.DataFrame]:
    valid = obs[SMALL_KEY].notna() & obs[MAJOR_KEY].notna() & obs[SAMPLE_KEY].notna()
    counts = obs.loc[valid, SMALL_KEY].astype(str).value_counts()
    keep_labels = counts[counts >= min_cells].index.tolist()
    valid &= obs[SMALL_KEY].astype(str).isin(keep_labels)

    rng = np.random.default_rng(seed)
    sampled_positions: list[int] = []
    small_to_major = []
    obs_valid = obs.loc[valid, [SMALL_KEY, MAJOR_KEY, SAMPLE_KEY]].copy()
    obs_valid["_pos"] = np.flatnonzero(valid.to_numpy())

    for small, group in obs_valid.groupby(SMALL_KEY, observed=True):
        positions = group["_pos"].to_numpy()
        if len(positions) > max_cells:
            positions = rng.choice(positions, size=max_cells, replace=False)
        sampled_positions.extend(positions.tolist())
        major = group[MAJOR_KEY].astype(str).value_counts().idxmax()
        small_to_major.append({"small_type": str(small), "major_type": str(major), "n_reference_cells_all": int(len(group))})

    sampled_positions = np.array(sorted(sampled_positions), dtype=int)
    mapping = pd.DataFrame(small_to_major).sort_values(["major_type", "small_type"]).reset_index(drop=True)
    return sampled_positions, mapping


def load_reference_subset(args: argparse.Namespace, spatial_genes: set[str]) -> tuple[ad.AnnData, pd.DataFrame]:
    common_genes = read_common_genes(args.common_genes)
    backed = sc.read_h5ad(args.reference_h5ad, backed="r")
    obs = backed.obs.copy()

    cell_positions, mapping = sample_reference_cells(
        obs,
        max_cells=args.max_cells_per_small,
        min_cells=args.min_cells_per_small,
        seed=args.seed,
    )

    gene_symbols = pd.Index(map(str, backed.var_names))
    gene_pool = spatial_genes & set(gene_symbols)
    if common_genes:
        gene_pool &= common_genes
    gene_mask = gene_symbols.isin(gene_pool)
    if gene_mask.sum() < 2000:
        raise RuntimeError(f"Too few shared genes for reference/spatial deconvolution: {gene_mask.sum()}")

    adata = backed[cell_positions, gene_mask].to_memory()
    backed.file.close()
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
        del adata.layers["counts"]
    adata.var_names = gene_symbols[gene_mask].tolist()
    adata.obs["small_type"] = adata.obs[SMALL_KEY].astype(str).astype("category")
    adata.obs["major_type"] = adata.obs[MAJOR_KEY].astype(str).astype("category")
    adata.obs["reference_sample"] = adata.obs[SAMPLE_KEY].astype(str).astype("category")
    adata = collapse_duplicate_genes(adata)
    adata.X = adata.X.astype(np.float32)
    return adata, mapping


def run_train(model, kwargs: dict) -> None:
    try:
        model.train(**kwargs)
    except TypeError as exc:
        if "accelerator" not in kwargs:
            raise
        fallback = dict(kwargs)
        fallback.pop("accelerator", None)
        fallback["use_gpu"] = kwargs.get("accelerator") == "gpu"
        print(f"[WARN] train() accelerator fallback after TypeError: {exc}")
        model.train(**fallback)


def export_posterior(model, adata: ad.AnnData, num_samples: int, batch_size, use_gpu: bool) -> ad.AnnData:
    sample_kwargs = {"num_samples": num_samples, "batch_size": batch_size, "accelerator": "gpu" if use_gpu else "cpu"}
    try:
        return model.export_posterior(adata, sample_kwargs=sample_kwargs)
    except TypeError as exc:
        sample_kwargs.pop("accelerator", None)
        sample_kwargs["use_gpu"] = use_gpu
        print(f"[WARN] export_posterior() fallback after TypeError: {exc}")
        return model.export_posterior(adata, sample_kwargs=sample_kwargs)


def mean_signature(adata: ad.AnnData) -> pd.DataFrame:
    labels = adata.obs["small_type"].astype(str)
    data = {}
    x = adata.X.tocsr() if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    for label in sorted(labels.unique()):
        mask = labels.eq(label).to_numpy()
        data[label] = np.asarray(x[mask].mean(axis=0)).ravel()
    sig = pd.DataFrame(data, index=adata.var_names)
    return sig.clip(lower=1e-6)


def build_or_load_signature(args: argparse.Namespace, spatial_genes: set[str], use_gpu: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    ref_dir = args.output_root / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    signature_csv = ref_dir / f"gse178341_{args.signature_method}_signature_small_type.csv"
    mapping_csv = ref_dir / "gse178341_small_to_major_mapping.csv"
    manifest_json = ref_dir / "reference_manifest.json"

    if signature_csv.exists() and mapping_csv.exists() and not args.force_reference:
        print(f"[INFO] Reusing cached signature: {signature_csv}")
        return pd.read_csv(signature_csv, index_col=0), pd.read_csv(mapping_csv)

    adata_ref, mapping = load_reference_subset(args, spatial_genes)
    print(f"[INFO] Reference subset: {adata_ref.n_obs} cells x {adata_ref.n_vars} genes x {adata_ref.obs['small_type'].nunique()} small types")

    try:
        selected = filter_genes(
            adata_ref,
            cell_count_cutoff=15,
            cell_percentage_cutoff2=0.03,
            nonz_mean_cutoff=1.12,
        )
        plt.savefig(ref_dir / "reference_gene_filter.png", dpi=180, bbox_inches="tight")
        plt.close("all")
        adata_ref = adata_ref[:, selected].copy()
    except Exception as exc:
        plt.close("all")
        print(f"[WARN] cell2location filter_genes failed, using all shared genes: {exc}")

    if args.signature_method == "mean":
        signature = mean_signature(adata_ref)
    else:
        RegressionModel.setup_anndata(
            adata=adata_ref,
            labels_key="small_type",
            batch_key="reference_sample",
        )
        ref_model = RegressionModel(adata_ref)
        run_train(
            ref_model,
            {
                "max_epochs": args.reference_epochs,
                "batch_size": args.reference_batch_size,
                "train_size": 1,
                "lr": 0.002,
                "accelerator": "gpu" if use_gpu else "cpu",
                "enable_progress_bar": False,
                "logger": False,
            },
        )
        adata_ref = export_posterior(
            ref_model,
            adata_ref,
            num_samples=args.posterior_samples,
            batch_size=args.reference_batch_size,
            use_gpu=use_gpu,
        )
        signature = adata_ref.varm["means_per_cluster_mu_fg"].copy()
        signature.columns = [str(c).replace("means_per_cluster_mu_fg_", "") for c in signature.columns]
        signature = signature.clip(lower=1e-6)

    signature.to_csv(signature_csv)
    mapping.to_csv(mapping_csv, index=False)
    write_json(
        manifest_json,
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "method": args.signature_method,
            "reference_h5ad": str(args.reference_h5ad),
            "common_genes": str(args.common_genes),
            "small_key": SMALL_KEY,
            "major_key": MAJOR_KEY,
            "max_cells_per_small": args.max_cells_per_small,
            "min_cells_per_small": args.min_cells_per_small,
            "reference_epochs": args.reference_epochs,
            "posterior_samples": args.posterior_samples,
            "n_reference_cells_used": int(adata_ref.n_obs),
            "n_signature_genes": int(signature.shape[0]),
            "n_small_types": int(signature.shape[1]),
        },
    )
    return signature, mapping


def read_positions(sample_dir: Path, obs_names: pd.Index) -> pd.DataFrame:
    pos_csv = sample_dir / "spatial" / "tissue_positions.csv"
    pos_list = sample_dir / "spatial" / "tissue_positions_list.csv"
    if pos_csv.exists():
        pos = pd.read_csv(pos_csv)
        if "barcode" not in pos.columns:
            pos.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_col_in_fullres", "pxl_row_in_fullres"]
    elif pos_list.exists():
        pos = pd.read_csv(pos_list, header=None)
        pos.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_col_in_fullres", "pxl_row_in_fullres"]
    else:
        return pd.DataFrame(index=obs_names)
    pos["barcode"] = pos["barcode"].astype(str)
    pos = pos.set_index("barcode").reindex(obs_names)
    return pos


def load_spatial_sample(sample_dir: Path, signature_genes: pd.Index) -> ad.AnnData:
    h5_path = next(sample_dir.glob("*filtered_feature_bc_matrix.h5"))
    adata = sc.read_10x_h5(h5_path)
    adata.var_names = pd.Index(map(str, adata.var_names))
    adata = collapse_duplicate_genes(adata)
    shared = [g for g in signature_genes if g in adata.var_names]
    if len(shared) < 2000:
        raise RuntimeError(f"{sample_dir.name}: only {len(shared)} signature genes found in spatial data")
    adata = adata[:, shared].copy()
    positions = read_positions(sample_dir, adata.obs_names)
    for col in positions.columns:
        adata.obs[col] = positions[col].values
    adata.obsm["spatial"] = positions[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(dtype=float)
    adata.obs["sample_id"] = sample_dir.name
    adata.X = adata.X.astype(np.float32)
    return adata


def clean_abundance_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).replace(prefix, "") for c in out.columns]
    return out


def fraction(df: pd.DataFrame) -> pd.DataFrame:
    denom = df.sum(axis=1).replace(0, np.nan)
    return df.div(denom, axis=0).fillna(0.0)


def aggregate_major(small: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    available = [c for c in small.columns if c in set(mapping["small_type"])]
    grouped = {}
    map_series = mapping.set_index("small_type")["major_type"].astype(str)
    for major in sorted(map_series.loc[available].unique()):
        cols = [c for c in available if map_series.loc[c] == major]
        grouped[major] = small[cols].sum(axis=1)
    return pd.DataFrame(grouped, index=small.index)


def run_sample(sample_dir: Path, signature: pd.DataFrame, mapping: pd.DataFrame, args: argparse.Namespace, use_gpu: bool) -> dict:
    out_dir = args.output_root / sample_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    done_file = out_dir / "cell2location_done.json"
    if done_file.exists() and not args.force:
        print(f"[INFO] Skip existing sample: {sample_dir.name}")
        return json.loads(done_file.read_text(encoding="utf-8"))

    adata_sp = load_spatial_sample(sample_dir, signature.index)
    sig = signature.loc[adata_sp.var_names].copy()
    Cell2location.setup_anndata(adata_sp)
    model = Cell2location(
        adata_sp,
        cell_state_df=sig,
        N_cells_per_location=args.n_cells_per_location,
        detection_alpha=args.detection_alpha,
    )
    run_train(
        model,
        {
            "max_epochs": args.spatial_epochs,
            "batch_size": min(args.spatial_batch_size, adata_sp.n_obs),
            "train_size": 1,
            "lr": 0.002,
            "accelerator": "gpu" if use_gpu else "cpu",
            "enable_progress_bar": False,
            "logger": False,
        },
    )
    adata_sp = export_posterior(
        model,
        adata_sp,
        num_samples=args.posterior_samples,
        batch_size=min(args.spatial_batch_size, adata_sp.n_obs),
        use_gpu=use_gpu,
    )

    small_mean = clean_abundance_columns(adata_sp.obsm["means_cell_abundance_w_sf"], "meanscell_abundance_w_sf_")
    small_q05 = clean_abundance_columns(adata_sp.obsm["q05_cell_abundance_w_sf"], "q05cell_abundance_w_sf_")
    small_q95 = clean_abundance_columns(adata_sp.obsm["q95_cell_abundance_w_sf"], "q95cell_abundance_w_sf_")
    major_mean = aggregate_major(small_mean, mapping)
    major_q05 = aggregate_major(small_q05, mapping)
    major_q95 = aggregate_major(small_q95, mapping)

    small_mean.to_csv(out_dir / "small_cell_abundance_means.csv")
    small_q05.to_csv(out_dir / "small_cell_abundance_q05.csv")
    small_q95.to_csv(out_dir / "small_cell_abundance_q95.csv")
    fraction(small_mean).to_csv(out_dir / "small_cell_fraction_means.csv")
    major_mean.to_csv(out_dir / "major_cell_abundance_means.csv")
    major_q05.to_csv(out_dir / "major_cell_abundance_q05.csv")
    major_q95.to_csv(out_dir / "major_cell_abundance_q95.csv")
    fraction(major_mean).to_csv(out_dir / "major_cell_fraction_means.csv")

    spot_meta = adata_sp.obs.copy()
    spot_meta.to_csv(out_dir / "spot_metadata.csv")
    adata_sp.write_h5ad(out_dir / "cell2location_spatial.h5ad", compression="gzip")

    payload = {
        "sample": sample_dir.name,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "n_spots": int(adata_sp.n_obs),
        "n_genes_used": int(adata_sp.n_vars),
        "n_small_types": int(small_mean.shape[1]),
        "n_major_types": int(major_mean.shape[1]),
        "mean_total_abundance": float(small_mean.sum(axis=1).mean()),
        "top_major_by_mean_abundance": major_mean.mean(axis=0).sort_values(ascending=False).head(5).to_dict(),
        "spatial_epochs": args.spatial_epochs,
        "posterior_samples": args.posterior_samples,
        "n_cells_per_location": args.n_cells_per_location,
        "detection_alpha": args.detection_alpha,
    }
    write_json(done_file, payload)
    write_json(out_dir / "run_manifest.json", payload)
    return payload


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    use_gpu = (not args.cpu) and torch.cuda.is_available()
    print(f"[INFO] use_gpu={use_gpu}")

    samples = list_spatial_samples(args.spatial_root, args.samples)
    if not samples:
        raise RuntimeError(f"No spatial samples found in {args.spatial_root}")
    spatial_genes = set()
    for sample_dir in samples:
        spatial_genes |= get_spatial_gene_symbols(sample_dir)
    print(f"[INFO] Spatial samples: {len(samples)}; union genes: {len(spatial_genes)}")

    signature, mapping = build_or_load_signature(args, spatial_genes, use_gpu)
    mapping = mapping[mapping["small_type"].isin(signature.columns)].copy()
    (args.output_root / "reference").mkdir(parents=True, exist_ok=True)
    mapping.to_csv(args.output_root / "reference" / "small_to_major_mapping_used.csv", index=False)

    results = []
    for i, sample_dir in enumerate(samples, 1):
        print(f"[INFO] [{i}/{len(samples)}] Running sample: {sample_dir.name}")
        result = run_sample(sample_dir, signature, mapping, args, use_gpu)
        results.append(result)
        print(
            "[INFO] Completed sample: {sample} spots={n_spots} genes={n_genes_used} "
            "small={n_small_types} major={n_major_types}".format(**result)
        )

    summary = pd.DataFrame(results)
    summary.to_csv(args.output_root / "cell2location_sample_summary.csv", index=False)
    write_json(
        args.output_root / "run_manifest.json",
        {
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "spatial_root": str(args.spatial_root),
            "output_root": str(args.output_root),
            "n_samples": len(samples),
            "samples": [p.name for p in samples],
            "signature_method": args.signature_method,
            "reference_epochs": args.reference_epochs,
            "spatial_epochs": args.spatial_epochs,
            "posterior_samples": args.posterior_samples,
            "use_gpu": use_gpu,
        },
    )
    print(f"[INFO] Wrote results to {args.output_root}")


if __name__ == "__main__":
    main()
