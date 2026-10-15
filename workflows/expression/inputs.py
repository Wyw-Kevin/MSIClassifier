"""Expression readers and feature definitions. No MSI labels or fitting."""
from pathlib import Path
import csv
import gzip
import numpy as np
import pandas as pd
from scipy import sparse

PANEL17 = ["cTNI14", "cTNI18", "cS18", "cTNI13", "cTNI16", "cTNI05",
           "cTNI07", "cS27", "cTNI04", "cS17", "cM01", "cP1", "cM05",
           "cM06", "cTNI08", "cTNI19", "cTNI10"]
PANEL12 = ["cS27", "cTNI16", "cM06", "cTNI05", "cTNI10", "cP1",
           "cTNI13", "cS18", "cTNI04", "cS17", "cM05", "cTNI08"]


def read_table(path):
    path = Path(path)
    sep = "\t" if ".tsv" in path.suffixes or ".txt" in path.suffixes else ","
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        header = next(csv.reader(handle, delimiter=sep))
    if len(header) != len(set(header)):
        raise ValueError("Duplicate column identifiers in input header")
    table = pd.read_csv(path, sep=sep, index_col=0)
    if table.empty or table.index.hasnans or not table.index.is_unique or not table.columns.is_unique:
        raise ValueError("Require a nonempty matrix with unique row and column identifiers")
    table.index = table.index.astype(str)
    table.columns = table.columns.astype(str)
    return table.astype(float)


def read_bulk(path):
    """Gene/probe rows by sample columns; no guessing of expression units."""
    return read_table(path)


def map_probes(table, mapping):
    required = ["probe_id", "gene_symbol"]
    if any(c not in mapping for c in required) or mapping[required].isna().any().any():
        raise ValueError("Probe mapping requires nonmissing probe_id and gene_symbol columns")
    mp = mapping[required].astype(str).drop_duplicates()
    if mp.probe_id.duplicated().any():
        raise ValueError("Each probe must map to one gene; resolve ambiguous probes before use")
    mp = mp.set_index("probe_id").gene_symbol
    selected = table.loc[table.index.isin(mp.index)].copy()
    if selected.empty:
        raise ValueError("No input probes overlap the supplied mapping")
    selected.index = mp.loc[selected.index].to_numpy()
    return selected.groupby(level=0, sort=False).median()


def bulk_scales(table, scale):
    x = table.to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Non-finite bulk expression; resolve missing genes before running")
    with np.errstate(over="ignore", invalid="ignore"):
        linear = x if scale == "linear" else np.exp2(x) - 1 if scale == "log2p1" else np.exp2(x) if scale == "log2" else None
    if linear is None:
        raise ValueError("Expression scale must be linear, log2 or log2p1")
    if not np.isfinite(linear).all() or (linear < 0).any():
        raise ValueError("Non-finite or negative linear expression; no clipping or shifting")
    if (linear >= 2**63 - 1024).any():
        raise ValueError("Expression exceeds safe int64 sampling range")
    cont = pd.DataFrame(linear, index=table.index, columns=table.columns)
    rounded = pd.DataFrame(np.rint(linear).astype(np.int64), index=table.index, columns=table.columns)
    log_values = pd.DataFrame(np.log2(linear + 1) if scale == "linear" else x, index=table.index, columns=table.columns)
    return cont, rounded, log_values


def read_cells(path, file_format="auto", transpose=False, layer=None):
    import anndata as ad
    path = Path(path)
    fmt = file_format
    if fmt == "auto":
        fmt = "10x-mtx" if path.is_dir() else "h5ad" if path.suffix == ".h5ad" else "10x-h5" if path.suffix == ".h5" else "loom" if path.suffix == ".loom" else "text"
    if fmt == "h5ad":
        a = ad.read_h5ad(path)
    elif fmt in {"10x-h5", "10x-mtx", "loom", "mtx"}:
        import scanpy as sc
        if fmt == "10x-h5":
            a = sc.read_10x_h5(path)
        elif fmt == "10x-mtx":
            a = sc.read_10x_mtx(path, var_names="gene_symbols", make_unique=False)
        elif fmt == "loom":
            a = sc.read_loom(path)
        else:
            raise ValueError("For standalone MTX attach gene/cell IDs in AnnData and save .h5ad, or use a 10x MTX directory")
    else:
        frame = read_table(path)
        if transpose:
            frame = frame.T
        a = ad.AnnData(sparse.csr_matrix(frame.to_numpy()), obs=pd.DataFrame(index=frame.index), var=pd.DataFrame(index=frame.columns))
    if layer is not None:
        if layer not in a.layers:
            raise ValueError("Requested expression layer is absent: " + layer)
        a.X = a.layers[layer].copy()
    if a.n_obs == 0 or a.n_vars == 0 or a.obs_names.hasnans or not a.obs_names.is_unique or not a.var_names.is_unique:
        raise ValueError("Require nonempty matrix with unique cell/spot IDs and gene symbols; resolve duplicate genes before normalization")
    a.X = sparse.csr_matrix(a.X, dtype=np.float32)
    if not np.isfinite(a.X.data).all() or (a.X.data < 0).any():
        raise ValueError("Cell/spot expression must be finite and nonnegative")
    return a


def require_counts(a):
    values = sparse.csr_matrix(a.X).data
    if not np.isfinite(values).all() or (values < 0).any() or not np.allclose(values, np.rint(values), rtol=0, atol=1e-6):
        raise ValueError("Raw counts must be finite nonnegative integer values")
    if (np.asarray(a.X.sum(axis=1)).ravel() <= 0).any():
        raise ValueError("Zero-library cells/spots must be removed explicitly before running")


def log_expression(a, scale):
    result = a.copy()
    if scale == "counts":
        require_counts(a)
        totals = np.asarray(result.X.sum(axis=1)).ravel()
        result.X = sparse.csr_matrix(result.X).multiply((10000. / totals)[:, None]).tocsr()
        result.X.data = np.log1p(result.X.data)
    elif scale != "log1p-cp10k":
        raise ValueError("Cell scale must be counts or log1p-cp10k; scaled/HVG/residual matrices are incompatible")
    return result


def parent(feature):
    for prefix, major in [("cTNI", "TNKILC"), ("cS", "Strom"), ("cM", "Myeloid"), ("cP", "Plasma")]:
        if feature.startswith(prefix):
            return major
    raise ValueError("Unknown subtype parent: " + feature)


def cell_fractions(cells, features, supported):
    missing = set(features) - set(supported)
    if missing:
        raise ValueError("unsupported reference subtypes: " + ", ".join(sorted(missing)))
    rows = []
    for sample, group in cells.groupby("sampleID", sort=False):
        row = {"sampleID": sample}
        for feature in features:
            par = group.major.eq(parent(feature))
            denominator = int(par.sum())
            row[feature] = float((par & group.subtype.eq(feature)).sum() / denominator) if denominator else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def weighted_mlh1(cells):
    rows = []
    for sample, group in cells.groupby("sampleID", sort=False):
        epi = group[group.is_epithelial]
        malignant = epi[epi.candidate]
        n, denominator = len(malignant), len(epi)
        mean = float(malignant.MLH1.mean()) if n else np.nan
        rows.append({"sampleID": sample, "n_malignant": n, "n_all_epithelial": denominator,
                     "malignant_MLH1_mean": mean,
                     "weighted_MLH1": mean * n / denominator if n and denominator else np.nan})
    return pd.DataFrame(rows)


def spatial_weighted_mlh1(allocated, mlh1_index, abundance):
    total = np.asarray(allocated.sum(axis=1)).ravel()
    gene = np.asarray(allocated[:, mlh1_index].toarray()).ravel()
    if (total <= 0).any():
        raise ValueError("Zero reconstructed candidate library; no proxy substitution")
    values = np.log1p(gene * 10000 / total)
    malignant = abundance["Epi_CNV_candidate"].to_numpy()
    denominator = float(abundance[["Epi_CNV_candidate", "Epi_other"]].to_numpy().sum())
    if denominator <= 0 or malignant.sum() <= 0:
        raise ValueError("No reconstructed epithelial/candidate abundance")
    return {"weighted_MLH1": float(values @ malignant / denominator),
            "spot_MLH1": values, "allocated_MLH1": gene, "allocated_library": total,
            "n_genes_normalization": allocated.shape[1]}
