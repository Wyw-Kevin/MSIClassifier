"""Run TOSICA migration and build scRNA MSI model input tables."""

# Packaging-only path resolution; archived originals retain exact source bytes.
from pathlib import Path as _MSIPackagePath
import sys as _MSIPackageSys
for _MSIPackageParent in _MSIPackagePath(__file__).resolve().parents:
    if (_MSIPackageParent / "project_paths.py").is_file():
        _MSIPackageSys.path.insert(0, str(_MSIPackageParent))
        break
from project_paths import project_root

import argparse
import json
import sys
import time
import types
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


PACKAGE_ROOT = project_root() / '最终模型/scRNA_model'
FINAL_MODEL_ROOT = PACKAGE_ROOT.parent
DEFAULT_INPUT_ROOT = FINAL_MODEL_ROOT / "data" / "scRNA" / "input"
DEFAULT_RESULT_ROOT = FINAL_MODEL_ROOT / "data" / "scRNA" / "result"
DEFAULT_REFERENCE_DIR = PACKAGE_ROOT / "reference"
DEFAULT_SELECTED_FEATURES = DEFAULT_REFERENCE_DIR / "selected_18_plus_mlh1_features.csv"
DEFAULT_SUBTYPE_FEATURES = DEFAULT_REFERENCE_DIR / "selected_18_subtype_features.csv"
DEFAULT_TOSICA_SOURCE = DEFAULT_REFERENCE_DIR / "TOSICA_source"

TableLike = Union[pd.DataFrame, str, Path]


DEFAULT_SUBTYPE_MODELS = {
    "TNKILC": {"dir": Path("CRC") / "TNKILC", "weight": "model-9.pth"},
    "Myeloid": {"dir": Path("CRC") / "Myeloid", "weight": "model-9.pth"},
    "Strom": {"dir": Path("CRC") / "Strom", "weight": "model-14.pth"},
    "Plasma": {"dir": Path("CRC") / "Plasma", "weight": "model-9.pth"},
}


def read_selected_features(selected_features=None):
    if selected_features is None:
        selected_features = DEFAULT_SELECTED_FEATURES
    if isinstance(selected_features, (str, Path)):
        table = pd.read_csv(Path(selected_features))
        if "feature" in table.columns:
            features = table["feature"].dropna().astype(str).tolist()
        else:
            features = table.iloc[:, 0].dropna().astype(str).tolist()
    else:
        features = [str(feature) for feature in selected_features]
    if not features:
        raise ValueError("No selected features were provided.")
    return features


def read_subtype_features(selected_features=None):
    features = read_selected_features(selected_features)
    return [feature for feature in features if feature != "mean_MLH1_epithelial"]


def _read_table(table):
    if isinstance(table, pd.DataFrame):
        return table.copy()
    return pd.read_csv(Path(table))


def _resolve_sample_col(frame, candidates):
    for candidate in candidates:
        if candidate and candidate in frame.columns:
            return str(candidate)
    raise ValueError("Could not find a sample column. Tried: %s" % ", ".join([str(c) for c in candidates if c]))


def read_label_dictionary(model_dir):
    table = pd.read_csv(Path(model_dir) / "label_dictionary.csv", index_col=0)
    if table.empty:
        raise ValueError("Label dictionary is empty: %s" % (Path(model_dir) / "label_dictionary.csv"))
    return table.iloc[:, 0].astype(str).tolist()


def read_train_genes(model_dir):
    """Read the exact model input gene order.

    CRC subtype projects store `train_genes.csv` without a header. Reading it
    with the default pandas header handling drops the first gene and shifts the
    TOSICA input tensor, so this function always uses `header=None` for that
    file.
    """
    model_dir = Path(model_dir)
    train_genes = model_dir / "train_genes.csv"
    if train_genes.exists():
        genes = pd.read_csv(train_genes, header=None).iloc[:, 0].dropna().astype(str).tolist()
        if genes and genes[0].lower() in {"gene", "genes"}:
            genes = genes[1:]
        if not genes:
            raise ValueError("No genes found in %s" % train_genes)
        return genes

    genes_used = model_dir / "genes_used.csv"
    if genes_used.exists():
        table = pd.read_csv(genes_used)
        if "gene" in table.columns:
            genes = table["gene"].dropna().astype(str).tolist()
        else:
            genes = table.iloc[:, 0].dropna().astype(str).tolist()
        if not genes:
            raise ValueError("No genes found in %s" % genes_used)
        return genes
    raise FileNotFoundError("No train_genes.csv or genes_used.csv found in %s" % model_dir)


def read_model_config(model_dir):
    model_dir = Path(model_dir)
    default = {"embed_dim": 48, "depth": 2, "num_heads": 4}
    manifest = model_dir / "model_manifest.json"
    if not manifest.exists():
        return default
    data = json.loads(manifest.read_text(encoding="utf-8"))
    cfg = data.get("config", {})
    return {
        "embed_dim": int(cfg.get("embed_dim", default["embed_dim"])),
        "depth": int(cfg.get("depth", default["depth"])),
        "num_heads": int(cfg.get("num_heads", default["num_heads"])),
    }


def install_tosica_source(tosica_source=None):
    tosica_source = Path(tosica_source or DEFAULT_TOSICA_SOURCE)
    if not (tosica_source / "TOSICA").exists():
        fallback = (project_root() / 'TOSICA')
        if (fallback / "TOSICA").exists():
            tosica_source = fallback
    if not (tosica_source / "TOSICA").exists():
        raise FileNotFoundError("TOSICA source package not found: %s" % tosica_source)
    tosica_pkg = types.ModuleType("TOSICA")
    tosica_pkg.__path__ = [str(tosica_source / "TOSICA")]
    sys.modules.setdefault("TOSICA", tosica_pkg)


def choose_device(name):
    import torch

    if str(name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(str(name))


def read_h5ad_dataset(h5ad_path, h5ad_backed="r", use_raw=False):
    import scanpy as sc

    adata = sc.read_h5ad(str(h5ad_path), backed=h5ad_backed)
    if not use_raw:
        return adata
    if adata.raw is None:
        if hasattr(adata, "file"):
            adata.file.close()
        raise ValueError("Requested --use-raw, but h5ad has no .raw matrix: %s" % h5ad_path)
    raw = adata.raw.to_adata()
    if hasattr(adata, "file"):
        adata.file.close()
    return raw


def load_torch_state(path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def logits_only(model, x):
    z = model.feature_embed(x)
    cls_token = model.cls_token.expand(z.shape[0], -1, -1)
    if model.dist_token is None:
        z = torch_cat_cls(x=None, cls_token=cls_token, z=z, dist_token=None)
    else:
        z = torch_cat_cls(x=None, cls_token=cls_token, z=z, dist_token=model.dist_token.expand(z.shape[0], -1, -1))
    for block in model.blocks:
        z, _ = block(z)
    z = model.norm(z)
    latent = model.pre_logits(z[:, 0])
    return model.head(latent)


def torch_cat_cls(x, cls_token, z, dist_token):
    import torch

    if dist_token is None:
        return torch.cat((cls_token, z), dim=1)
    return torch.cat((cls_token, dist_token, z), dim=1)


class GenePlan:
    """Map h5ad columns into the exact gene order expected by a TOSICA model."""

    def __init__(self, adata_var_names, model_genes):
        first = {}
        for idx, gene in enumerate(pd.Index(adata_var_names).astype(str)):
            if str(gene) not in first:
                first[str(gene)] = int(idx)
        present = [(model_pos, first[gene]) for model_pos, gene in enumerate(model_genes) if gene in first]
        self.model_genes = list(model_genes)
        self.n_genes = len(model_genes)
        self.missing_genes = [gene for gene in model_genes if gene not in first]
        present_sorted = sorted(present, key=lambda item: item[1])
        self.model_positions = np.asarray([item[0] for item in present_sorted], dtype=np.int64)
        self.adata_positions = np.asarray([item[1] for item in present_sorted], dtype=np.int64)

    def load_batch(self, adata, row_idx):
        from scipy import sparse

        out = np.zeros((len(row_idx), self.n_genes), dtype=np.float32)
        if len(self.adata_positions) == 0:
            return out
        x = adata[row_idx, self.adata_positions].X
        if sparse.issparse(x):
            x = x.toarray()
        else:
            x = np.asarray(x)
        out[:, self.model_positions] = x.astype(np.float32, copy=False)
        return out


def load_tosica_model(model_dir, weight_name, device, tosica_source=None):
    install_tosica_source(tosica_source)
    from TOSICA.TOSICA_model import scTrans_model as create_model

    import torch

    model_dir = Path(model_dir)
    labels = read_label_dictionary(model_dir)
    genes = read_train_genes(model_dir)
    cfg = read_model_config(model_dir)
    mask = np.load(model_dir / "mask.npy")
    model = create_model(
        num_classes=len(labels),
        num_genes=len(genes),
        mask=mask,
        embed_dim=cfg["embed_dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        has_logits=False,
    )
    state = load_torch_state(model_dir / weight_name)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, labels, genes, cfg


def predict_model(adata, row_idx, model_dir, weight_name, batch_size, device, model_name, tosica_source=None):
    import torch

    model, labels, genes, cfg = load_tosica_model(model_dir, weight_name, device, tosica_source=tosica_source)
    plan = GenePlan(pd.Index(adata.var_names), genes)
    pred_idx = np.zeros(len(row_idx), dtype=np.int64)
    pred_prob = np.zeros(len(row_idx), dtype=np.float32)
    start_time = time.time()
    with torch.no_grad():
        for start in range(0, len(row_idx), batch_size):
            end = min(start + batch_size, len(row_idx))
            rows = row_idx[start:end]
            x = plan.load_batch(adata, rows)
            xb = torch.from_numpy(x).to(device)
            logits = logits_only(model, xb)
            probs = torch.softmax(logits, dim=1)
            values, idx = probs.max(dim=1)
            pred_idx[start:end] = idx.detach().cpu().numpy()
            pred_prob[start:end] = values.detach().cpu().numpy()
            if end == len(row_idx) or end % (batch_size * 10) == 0:
                print("%s: %d/%d" % (model_name, end, len(row_idx)), flush=True)
    pred_label = np.asarray(labels, dtype=object)[pred_idx]
    info = {
        "model": model_name,
        "model_dir": str(model_dir),
        "weight": weight_name,
        "n_cells_predicted": int(len(row_idx)),
        "n_classes": int(len(labels)),
        "n_genes": int(len(genes)),
        "embed_dim": int(cfg["embed_dim"]),
        "depth": int(cfg["depth"]),
        "num_heads": int(cfg["num_heads"]),
        "missing_genes": int(len(plan.missing_genes)),
        "missing_gene_examples": plan.missing_genes[:20],
        "elapsed_sec": round(time.time() - start_time, 2),
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pred_label, pred_prob, labels, genes, info


def model_set_config(reference_dir=None):
    reference_dir = Path(reference_dir or DEFAULT_REFERENCE_DIR)
    subtypes = {}
    for major, cfg in DEFAULT_SUBTYPE_MODELS.items():
        subtypes[major] = {"dir": reference_dir / cfg["dir"], "weight": cfg["weight"]}
    return {
        "prefix": None,
        "major": {"dir": reference_dir / "tosica_major", "weight": "model_best.pth"},
        "subtypes": subtypes,
    }


def parent_for_feature(feature):
    feature = str(feature)
    if feature.startswith("cTNI"):
        return "TNKILC"
    if feature.startswith("cM"):
        return "Myeloid"
    if feature.startswith("cS"):
        return "Strom"
    if feature.startswith("cP"):
        return "Plasma"
    raise ValueError("Unknown feature parent for %s" % feature)


def read_obs_table(adata, n_obs, sample_col=None, tumor_col="Class"):
    obs = adata.obs.iloc[:n_obs].copy().reset_index(drop=True)
    obs.insert(0, "cell_id", pd.Index(adata.obs_names[:n_obs]).astype(str))
    resolved_sample_col = _resolve_sample_col(obs, [sample_col, "Sample", "sample", "sampleID", "SampleID"])
    if resolved_sample_col != "Sample":
        obs = obs.rename(columns={resolved_sample_col: "Sample"})
    if tumor_col not in obs.columns:
        raise ValueError("h5ad obs must contain tumor/cell class column: %s" % tumor_col)
    if tumor_col != "Class":
        obs = obs.rename(columns={tumor_col: "Class"})
    for col in obs.columns:
        obs[col] = obs[col].astype(str)
    if "MMR" not in obs.columns and "MSI_status" in obs.columns:
        status = obs["MSI_status"].str.upper()
        obs["MMR"] = np.where(status.str.contains("MSI"), "MSI", np.where(status.str.contains("MSS"), "MSS", ""))
    if "MMR" not in obs.columns and "MSI_MSS" in obs.columns:
        status = obs["MSI_MSS"].str.upper()
        obs["MMR"] = np.where(status.eq("MSI"), "MSI", np.where(status.eq("MSS"), "MSS", ""))
    return obs


def reuse_major_predictions(cells, major_cells_path):
    major_cells_path = Path(major_cells_path)
    major_cells = pd.read_csv(major_cells_path)
    if len(major_cells) != len(cells):
        raise ValueError(
            "Cannot reuse major predictions: %s has %d cells, but current h5ad slice has %d cells"
            % (major_cells_path, len(major_cells), len(cells))
        )
    if "cell_id" in major_cells.columns and not cells["cell_id"].astype(str).equals(major_cells["cell_id"].astype(str)):
        raise ValueError("Cannot reuse major predictions: cell_id order differs in %s" % major_cells_path)
    for column in ["major_prediction", "major_probability"]:
        if column not in major_cells.columns:
            raise ValueError("Cannot reuse major predictions: %s is missing from %s" % (column, major_cells_path))
    cells["major_prediction"] = major_cells["major_prediction"].astype(str).to_numpy()
    cells["major_probability"] = pd.to_numeric(major_cells["major_probability"], errors="coerce").to_numpy()
    return cells, {
        "model": "Major",
        "model_dir": str(major_cells_path),
        "weight": "reused_existing_predictions",
        "n_cells_predicted": int(len(cells)),
        "reuse_source": str(major_cells_path),
    }


def mean_gene_by_mask(adata, gene, sample_values, sample_order, mask):
    from scipy import sparse

    var_names = pd.Index(adata.var_names).astype(str)
    if gene not in set(var_names):
        return pd.DataFrame({"sampleID": sample_order, "mean_MLH1_epithelial": 0.0, "n_epithelial_cells_for_MLH1": 0})
    gene_idx = int(np.where(var_names.to_numpy() == gene)[0][0])
    rows = np.where(mask)[0]
    sums = {sample: 0.0 for sample in sample_order}
    counts = {sample: 0 for sample in sample_order}
    batch = 4096
    for start in range(0, len(rows), batch):
        end = min(start + batch, len(rows))
        row_batch = rows[start:end]
        x = adata[row_batch, [gene_idx]].X
        if sparse.issparse(x):
            values = np.asarray(x.toarray()).ravel()
        else:
            values = np.asarray(x).ravel()
        samples = sample_values[row_batch]
        for sample, value in zip(samples, values):
            sample = str(sample)
            sums[sample] = sums.get(sample, 0.0) + float(value)
            counts[sample] = counts.get(sample, 0) + 1
    out = []
    for sample in sample_order:
        n = counts.get(sample, 0)
        out.append(
            {
                "sampleID": sample,
                "mean_MLH1_epithelial": float(sums.get(sample, 0.0) / n) if n else 0.0,
                "n_epithelial_cells_for_MLH1": int(n),
            }
        )
    return pd.DataFrame(out)


def load_reference_mlh1(sample_order, reference_mlh1_path=None, reference_mlh1_fallback_path=None):
    out = pd.DataFrame({"sampleID": sample_order})
    if reference_mlh1_path is not None and Path(reference_mlh1_path).exists():
        ref = pd.read_csv(reference_mlh1_path)
        ref["sampleID"] = ref["sampleID"].astype(str)
        ref = ref.rename(
            columns={
                "mean_MLH1_malignant": "reference_mean_MLH1_epithelial",
                "n_malignant_cells": "reference_n_epithelial_cells_for_MLH1",
                "mean_MLH1_epithelial": "reference_mean_MLH1_epithelial",
                "n_epithelial_cells_for_MLH1": "reference_n_epithelial_cells_for_MLH1",
            }
        )
        keep = ["sampleID"]
        for col in ["reference_mean_MLH1_epithelial", "reference_n_epithelial_cells_for_MLH1"]:
            if col in ref.columns and col not in keep:
                keep.append(col)
        out = out.merge(ref[keep].drop_duplicates("sampleID", keep="first"), on="sampleID", how="left")
    if reference_mlh1_fallback_path is not None and Path(reference_mlh1_fallback_path).exists():
        fallback = pd.read_csv(reference_mlh1_fallback_path)
        fallback["sampleID"] = fallback["sampleID"].astype(str)
        fallback = fallback.rename(columns={"mean_MLH1_epithelial": "fallback_mean_MLH1_epithelial"})
        if "fallback_mean_MLH1_epithelial" in fallback.columns:
            out = out.merge(
                fallback[["sampleID", "fallback_mean_MLH1_epithelial"]].drop_duplicates("sampleID", keep="first"),
                on="sampleID",
                how="left",
            )
    for column in ["reference_mean_MLH1_epithelial", "reference_n_epithelial_cells_for_MLH1", "fallback_mean_MLH1_epithelial"]:
        if column not in out.columns:
            out[column] = np.nan
    return out


def sample_labels_from_cells(cells, tumor_mask):
    if "MMR" not in cells.columns:
        return pd.DataFrame(columns=["sampleID", "MMR"])
    labels = cells.loc[tumor_mask, ["Sample", "MMR"]].copy()
    labels["Sample"] = labels["Sample"].astype(str)
    labels["MMR"] = labels["MMR"].astype(str)
    labels = labels[~labels["MMR"].str.lower().isin(["", "nan", "none", "null"])]
    if labels.empty:
        return pd.DataFrame(columns=["sampleID", "MMR"])
    labels = (
        labels.groupby("Sample", sort=False)["MMR"]
        .agg(lambda values: values.value_counts().index[0])
        .reset_index()
        .rename(columns={"Sample": "sampleID"})
    )
    return labels


def merge_mmr_labels(feature_rows, labels):
    if labels is None or labels.empty or "MMR" not in labels.columns:
        return feature_rows
    labels = labels[["sampleID", "MMR"]].copy()
    labels["sampleID"] = labels["sampleID"].astype(str)
    labels["MMR"] = labels["MMR"].astype(str)
    labels = labels[~labels["MMR"].str.lower().isin(["", "nan", "none", "null"])]
    if labels.empty:
        return feature_rows
    labels = labels.drop_duplicates("sampleID", keep="first")
    if "MMR" not in feature_rows.columns:
        return feature_rows.merge(labels, on="sampleID", how="left")
    merged = feature_rows.merge(labels.rename(columns={"MMR": "MMR_new"}), on="sampleID", how="left")
    merged["MMR"] = merged["MMR_new"].where(merged["MMR_new"].notna(), merged["MMR"])
    return merged.drop(columns=["MMR_new"])


def build_feature_matrix(
    adata,
    cells,
    out_dir,
    prefix,
    dataset,
    selected_features=None,
    known_features_path=None,
    reference_mlh1_path=None,
    reference_mlh1_fallback_path=None,
    tumor_value="Tumor",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_features = read_subtype_features(selected_features or DEFAULT_SUBTYPE_FEATURES)
    cells = cells.copy()
    tumor = cells["Class"].astype(str).eq(str(tumor_value)).to_numpy()
    sample_values = cells["Sample"].astype(str).to_numpy()
    sample_order = sorted(cells.loc[tumor, "Sample"].astype(str).unique().tolist())
    feature_rows = pd.DataFrame({"dataset": str(dataset), "sampleID": sample_order})
    feature_rows = merge_mmr_labels(feature_rows, sample_labels_from_cells(cells, tumor))

    for feature in selected_features:
        parent = parent_for_feature(feature)
        values = []
        for sample in sample_order:
            sample_mask = tumor & (sample_values == sample) & cells["major_prediction"].astype(str).eq(parent).to_numpy()
            denom = int(sample_mask.sum())
            if denom == 0:
                values.append(0.0)
                continue
            child_mask = sample_mask & cells["final_migrated_subtype"].astype(str).eq(feature).to_numpy()
            values.append(float(child_mask.sum() / float(denom)))
        feature_rows[feature] = values

    mlh1 = mean_gene_by_mask(
        adata,
        gene="MLH1",
        sample_values=sample_values,
        sample_order=sample_order,
        mask=tumor & cells["major_prediction"].astype(str).eq("Epi").to_numpy(),
    )
    ref_mlh1 = load_reference_mlh1(sample_order, reference_mlh1_path, reference_mlh1_fallback_path)
    feature_rows = feature_rows.merge(mlh1, on="sampleID", how="left")
    feature_rows = feature_rows.merge(ref_mlh1, on="sampleID", how="left")
    feature_rows["mean_MLH1_epithelial"] = feature_rows["reference_mean_MLH1_epithelial"].where(
        feature_rows["reference_mean_MLH1_epithelial"].notna(),
        feature_rows["fallback_mean_MLH1_epithelial"].where(
            feature_rows["fallback_mean_MLH1_epithelial"].notna(),
            feature_rows["mean_MLH1_epithelial"],
        ),
    )
    feature_rows["n_epithelial_cells_for_MLH1"] = feature_rows["reference_n_epithelial_cells_for_MLH1"].where(
        feature_rows["reference_n_epithelial_cells_for_MLH1"].notna(),
        feature_rows["n_epithelial_cells_for_MLH1"],
    )
    feature_rows = feature_rows.drop(
        columns=[
            "reference_mean_MLH1_epithelial",
            "reference_n_epithelial_cells_for_MLH1",
            "fallback_mean_MLH1_epithelial",
        ],
        errors="ignore",
    )

    if known_features_path is not None and Path(known_features_path).exists():
        known = pd.read_csv(known_features_path)
        known["sampleID"] = known["sampleID"].astype(str)
        if "MMR" in known.columns:
            feature_rows = merge_mmr_labels(feature_rows, known[["sampleID", "MMR"]])

    meta_cols = ["dataset", "sampleID"]
    if "MMR" in feature_rows.columns and feature_rows["MMR"].notna().any():
        meta_cols.append("MMR")
    output_cols = meta_cols + selected_features + ["mean_MLH1_epithelial", "n_epithelial_cells_for_MLH1"]
    feature_rows = feature_rows.reindex(columns=output_cols)
    feature_rows.to_csv(out_dir / ("%s_processed.csv" % prefix), index=False)
    return feature_rows


def _default_known_features(dataset, input_root):
    candidates = [
        Path(input_root) / ("%s_18_plus_mlh1_features.csv" % str(dataset).lower()),
        Path(input_root) / ("%s_18_plus_mlh1_features.csv" % str(dataset)),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _default_mlh1_reference(dataset, input_root):
    candidates = [
        Path(input_root) / ("%s_epithelial_mlh1_by_sample.csv" % str(dataset).lower()),
        Path(input_root) / ("%s_epithelial_mlh1_by_sample.csv" % str(dataset)),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _write_counts(cells, out_dir, prefix):
    major_counts = cells.groupby(["Class", "Sample", "major_prediction"], dropna=False).size().reset_index(name="n_cells")
    major_counts.to_csv(Path(out_dir) / ("%s_tosica_major_counts_by_sample.csv" % prefix), index=False)
    subtype_counts = (
        cells[cells["subtype_model"].ne("")]
        .groupby(["Class", "Sample", "subtype_model", "subtype_prediction"], dropna=False)
        .size()
        .reset_index(name="n_cells")
    )
    subtype_counts.to_csv(Path(out_dir) / ("%s_tosica_subtype_counts_by_sample.csv" % prefix), index=False)


def process_scrna_h5ad(
    dataset,
    h5ad_path,
    input_root=DEFAULT_INPUT_ROOT,
    result_root=DEFAULT_RESULT_ROOT,
    reference_dir=DEFAULT_REFERENCE_DIR,
    selected_features=None,
    known_features_path=None,
    reference_mlh1_path=None,
    reference_mlh1_fallback_path=None,
    major_cells_path=None,
    batch_size=1024,
    limit_cells=None,
    device="cuda:0",
    sample_col=None,
    tumor_col="Class",
    tumor_value="Tumor",
    tosica_source=DEFAULT_TOSICA_SOURCE,
    prefix=None,
    h5ad_backed="r",
    use_raw=False,
    min_total_cells=200,
):
    """Run the packaged hierarchical migration and write `<dataset>_processed.csv`.

    The default model set uses the final `tosica_major` model for major cell
    classes and the CRC subtype checkpoints for TNKILC/Myeloid/Strom/Plasma.
    CRC subtype inputs are aligned to each checkpoint's `train_genes.csv` order
    by `GenePlan`.
    """
    dataset = str(dataset)
    prefix = str(prefix or dataset)
    process_dir = Path(result_root) / "process"
    process_dir.mkdir(parents=True, exist_ok=True)
    cfg = model_set_config(reference_dir)
    device_obj = choose_device(device)
    adata = read_h5ad_dataset(h5ad_path, h5ad_backed=h5ad_backed, use_raw=use_raw)
    try:
        n_obs = adata.n_obs if limit_cells is None else min(int(limit_cells), adata.n_obs)
        all_rows = np.arange(n_obs, dtype=np.int64)
        cells = read_obs_table(adata, n_obs, sample_col=sample_col, tumor_col=tumor_col)
        sample_counts = cells.groupby("Sample", sort=False).size()
        eligible_samples = set(sample_counts[sample_counts >= int(min_total_cells)].index.astype(str))
        eligible_rows = np.flatnonzero(cells["Sample"].astype(str).isin(eligible_samples).to_numpy())
        if len(eligible_rows) != len(cells):
            filtered = adata[eligible_rows].copy()
            if hasattr(adata, "file"):
                adata.file.close()
            adata = filtered
            n_obs = adata.n_obs
            all_rows = np.arange(n_obs, dtype=np.int64)
            cells = read_obs_table(adata, n_obs, sample_col=sample_col, tumor_col=tumor_col)

        selected_models = [
            {"model": "Major", "model_dir": str(cfg["major"]["dir"]), "weight": cfg["major"]["weight"]},
            {"model": "TNKILC", "model_dir": str(cfg["subtypes"]["TNKILC"]["dir"]), "weight": cfg["subtypes"]["TNKILC"]["weight"]},
            {"model": "Myeloid", "model_dir": str(cfg["subtypes"]["Myeloid"]["dir"]), "weight": cfg["subtypes"]["Myeloid"]["weight"]},
            {"model": "Strom", "model_dir": str(cfg["subtypes"]["Strom"]["dir"]), "weight": cfg["subtypes"]["Strom"]["weight"]},
            {"model": "Plasma", "model_dir": str(cfg["subtypes"]["Plasma"]["dir"]), "weight": cfg["subtypes"]["Plasma"]["weight"]},
        ]
        if major_cells_path is not None:
            selected_models[0]["reuse_source"] = str(major_cells_path)
        pd.DataFrame(selected_models).to_csv(process_dir / ("%s_selected_migration_models.csv" % prefix), index=False)

        if major_cells_path is not None:
            cells, major_info = reuse_major_predictions(cells, major_cells_path)
        else:
            major_label, major_prob, _, _, major_info = predict_model(
                adata,
                all_rows,
                cfg["major"]["dir"],
                cfg["major"]["weight"],
                batch_size,
                device_obj,
                "Major",
                tosica_source=tosica_source,
            )
            cells["major_prediction"] = major_label
            cells["major_probability"] = major_prob

        cells["subtype_model"] = ""
        cells["subtype_prediction"] = ""
        cells["subtype_probability"] = np.nan
        cells["final_migrated_subtype"] = ""
        model_infos = [major_info]

        for major, subtype_cfg in cfg["subtypes"].items():
            selected = np.where(cells["major_prediction"].to_numpy() == major)[0].astype(np.int64)
            if len(selected) == 0:
                model_infos.append(
                    {
                        "model": major,
                        "model_dir": str(subtype_cfg["dir"]),
                        "weight": subtype_cfg["weight"],
                        "n_cells_predicted": 0,
                    }
                )
                continue
            subtype_label, subtype_prob, _, _, info = predict_model(
                adata,
                selected,
                subtype_cfg["dir"],
                subtype_cfg["weight"],
                batch_size,
                device_obj,
                major,
                tosica_source=tosica_source,
            )
            cells.loc[selected, "subtype_model"] = major
            cells.loc[selected, "subtype_prediction"] = subtype_label
            cells.loc[selected, "subtype_probability"] = subtype_prob
            cells.loc[selected, "final_migrated_subtype"] = subtype_label
            model_infos.append(info)

        cells.to_csv(process_dir / ("%s_tosica_migration_cells.csv" % prefix), index=False)
        _write_counts(cells, process_dir, prefix)
        if known_features_path is None:
            known_features_path = _default_known_features(dataset, input_root)
        if reference_mlh1_path is None:
            reference_mlh1_path = _default_mlh1_reference(dataset, input_root)
        features = build_feature_matrix(
            adata,
            cells,
            out_dir=process_dir,
            prefix=prefix,
            dataset=dataset,
            selected_features=selected_features,
            known_features_path=known_features_path,
            reference_mlh1_path=reference_mlh1_path,
            reference_mlh1_fallback_path=reference_mlh1_fallback_path,
            tumor_value=tumor_value,
        )
    finally:
        if hasattr(adata, "file"):
            adata.file.close()

    manifest = {
        "h5ad": str(h5ad_path),
        "dataset": dataset,
        "result_root": str(result_root),
        "reference_dir": str(reference_dir),
        "prefix": prefix,
        "n_cells": int(n_obs),
        "sample_qc": {"rule": "total_cells >= %d" % int(min_total_cells),
                      "min_total_cells_inclusive": int(min_total_cells),
                      "n_samples_after_qc": int(cells["Sample"].nunique()),
                      "n_tumor_samples_after_qc": int(cells.loc[cells["Class"].eq(tumor_value), "Sample"].nunique())},
        "batch_size": int(batch_size),
        "device": str(device_obj),
        "h5ad_backed": h5ad_backed,
        "use_raw": bool(use_raw),
        "selected_models": selected_models,
        "model_infos": model_infos,
        "processed_path": str(process_dir / ("%s_processed.csv" % prefix)),
        "n_feature_samples": int(features.shape[0]),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (process_dir / ("%s_migration_manifest.json" % prefix)).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return process_dir / ("%s_processed.csv" % prefix)


def process_scrna_feature_table(
    dataset,
    feature_table,
    result_root=DEFAULT_RESULT_ROOT,
    selected_features=None,
    sample_col="sampleID",
    label_col="MMR",
):
    """Post-process an existing sample-level feature table into model input."""
    dataset = str(dataset)
    features = read_selected_features(selected_features)
    table = _read_table(feature_table)
    source_sample_col = _resolve_sample_col(table, [sample_col, "sampleID", "SampleID", "sample", "Sample"])
    if source_sample_col != sample_col:
        table = table.rename(columns={source_sample_col: sample_col})
    if "dataset" not in table.columns:
        table.insert(0, "dataset", dataset)
    missing = [feature for feature in features if feature not in table.columns]
    if missing:
        raise ValueError("Feature table is missing model features: %s" % missing)
    meta_cols = ["dataset", sample_col]
    if label_col in table.columns and table[label_col].notna().any():
        meta_cols.append(label_col)
    extra_cols = [col for col in ["n_epithelial_cells_for_MLH1"] if col in table.columns]
    output = table[meta_cols + features + extra_cols].copy()
    output_dir = Path(result_root) / "process"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / ("%s_processed.csv" % dataset)
    output.to_csv(output_path, index=False)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--h5ad", type=Path, default=None)
    parser.add_argument("--feature-table", type=Path, default=None)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--selected-features", type=Path, default=DEFAULT_SELECTED_FEATURES)
    parser.add_argument("--known-features-path", type=Path, default=None)
    parser.add_argument("--reference-mlh1-path", type=Path, default=None)
    parser.add_argument("--reference-mlh1-fallback-path", type=Path, default=None)
    parser.add_argument("--major-cells", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--limit-cells", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample-col", default=None)
    parser.add_argument("--tumor-col", default="Class")
    parser.add_argument("--tumor-value", default="Tumor")
    parser.add_argument("--tosica-source", type=Path, default=DEFAULT_TOSICA_SOURCE)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--in-memory", action="store_true")
    parser.add_argument("--use-raw", action="store_true")
    parser.add_argument("--min-total-cells", type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.feature_table is not None:
        path = process_scrna_feature_table(
            args.dataset,
            args.feature_table,
            result_root=args.result_root,
            selected_features=args.selected_features,
        )
    elif args.h5ad is not None:
        path = process_scrna_h5ad(
            args.dataset,
            args.h5ad,
            input_root=args.input_root,
            result_root=args.result_root,
            reference_dir=args.reference_dir,
            selected_features=args.selected_features,
            known_features_path=args.known_features_path,
            reference_mlh1_path=args.reference_mlh1_path,
            reference_mlh1_fallback_path=args.reference_mlh1_fallback_path,
            major_cells_path=args.major_cells,
            batch_size=args.batch_size,
            limit_cells=args.limit_cells,
            device=args.device,
            sample_col=args.sample_col,
            tumor_col=args.tumor_col,
            tumor_value=args.tumor_value,
            tosica_source=args.tosica_source,
            prefix=args.prefix,
            h5ad_backed=None if args.in_memory else "r",
            use_raw=args.use_raw,
            min_total_cells=args.min_total_cells,
        )
    else:
        raise SystemExit("Either --h5ad or --feature-table is required.")
    print(json.dumps({"processed_path": str(path)}, indent=2))


if __name__ == "__main__":
    main()
