"""Independent annotation and stromal-only CNV, then weighted frozen scoring."""
import json
import random
import numpy as np
import pandas as pd
from inputs import PANEL18, cell_fractions, weighted_mlh1
from runners import require_resources, score_features, manifest
import tosica


def infer_cnv(a, major, gene_positions, out):
    import infercnvpy as cnv
    if not major.eq("Strom").any():
        raise ValueError("No frozen TOSICA Strom reference; stop CNV analysis")
    rng = np.random.default_rng(2026)
    tumor = np.flatnonzero(a.obs.is_tumor.to_numpy())
    reference = np.flatnonzero(major.eq("Strom").to_numpy())
    if len(tumor) > 100000:
        tumor = rng.choice(tumor, 100000, replace=False)
    if len(reference) > 20000:
        reference = rng.choice(reference, 20000, replace=False)
    selected = np.union1d(tumor, reference)
    positions = pd.read_csv(gene_positions, sep="\t", header=None, names=["gene", "chromosome", "start", "end"])
    positions = positions.drop_duplicates("gene").set_index("gene")
    shared = a.var_names.intersection(positions.index)
    if len(shared) < 2000:
        raise ValueError("Fewer than 2000 genes have genomic positions")
    x = a[selected, shared].copy()
    x.var[["chromosome", "start", "end"]] = positions.loc[shared, ["chromosome", "start", "end"]].to_numpy()
    x.obs["cnv_reference"] = np.where(major.iloc[selected].eq("Strom"), "stromal", "")
    cnv.tl.infercnv(x, reference_key="cnv_reference", reference_cat=["stromal"], lfc_clip=3, window_size=100,
                   step=10, dynamic_threshold=1.5, exclude_chromosomes=("chrX", "chrY"), chunksize=5000,
                   n_jobs=4, key_added="cnv", calculate_gene_values=False)
    cnv.tl.pca(x, use_rep="cnv", key_added="cnv_pca", random_state=2026)
    cnv.pp.neighbors(x, use_rep="cnv_pca", key_added="cnv_neighbors", random_state=2026)
    cnv.tl.leiden(x, neighbors_key="cnv_neighbors", key_added="cnv_leiden", random_state=2026)
    cnv.tl.cnv_score(x, groupby="cnv_leiden", use_rep="cnv", key_added="cnv_score")
    obs = x.obs[["sampleID", "is_tumor", "cnv_reference", "cnv_leiden", "cnv_score"]].copy()
    obs["is_epithelial"] = major.iloc[selected].eq("Epi").to_numpy()
    threshold = float(obs.loc[obs.cnv_reference.eq("stromal"), "cnv_score"].quantile(.95))
    if not np.isfinite(threshold):
        raise ValueError("Non-finite stromal-reference q95 threshold")
    obs["candidate"] = (obs.is_tumor & obs.is_epithelial & obs.cnv_reference.eq("") &
                        obs.cnv_score.ge(threshold - 1e-12))
    obs["MLH1"] = a[selected, ["MLH1"]].X.toarray().ravel()
    obs.to_csv(out / "cnv_retained_cells.csv.gz", index_label="cell_id")
    (out / "cnv_definition.json").write_text(json.dumps({"seed": 2026, "reference": "stromal_only", "q": .95,
        "threshold": threshold, "n_retained": len(obs), "study_defined_threshold": True,
        "sampling": "up to100000 tumor-tissue cells plus up to20000 Strom reference cells; union without duplicate cell IDs",
        "expression": "log1p full-library CP10k, before genomic-position subsetting",
        "parameters": {"lfc_clip": 3, "window_size": 100, "step": 10, "dynamic_threshold": 1.5, "n_jobs": 4}}, indent=2))
    return weighted_mlh1(obs[obs.is_tumor])


def run(args, a, out):
    import torch
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    records = require_resources(args.resource_dir, ["ecology/", "cnv_major/", "TOSICA_source/", "gene_positions.tsv"])
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; select --device cpu")
    random.seed(2026); np.random.seed(2026); torch.manual_seed(2026)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2026)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    counts = a.obs.groupby("sampleID", sort=False).size()
    a = a[a.obs.sampleID.isin(counts[counts >= 200].index)].copy()
    if a.n_obs == 0 or not a.obs.is_tumor.any():
        raise ValueError("No tumor samples after total-cell QC >=200")
    if "MLH1" not in a.var_names:
        raise ValueError("MLH1 is unmeasured")
    r = args.resource_dir
    models = {"TNKILC": ("CRC/TNKILC", "model-9.pth"), "Myeloid": ("CRC/Myeloid", "model-9.pth"),
              "Strom": ("CRC/Strom", "model-14.pth"), "Plasma": ("CRC/Plasma", "model-9.pth")}
    supported = set().union(*(set(tosica.labels(r / "ecology" / folder)) for folder, _ in models.values()))
    missing = set(PANEL18) - supported
    if missing:
        raise ValueError("Unmeasured fixed18 checkpoint classes: " + ", ".join(sorted(missing)))
    rows = np.arange(a.n_obs)
    if args.dataset == "GSE178341":
        if any(c not in a.obs for c in ["clTopLevel", "cl295v11SubShort"]):
            raise ValueError("GSE178341 requires its original clTopLevel and cl295v11SubShort definitions")
        major = a.obs.clTopLevel.astype(str).to_numpy()
        subtype = a.obs.cl295v11SubShort.astype(str).to_numpy()
        cnv_major = pd.Series(major, index=a.obs_names)
        infos = [{"annotation": "original fixed GSE178341 reference definitions"}]
    else:
        major, probability, info = tosica.predict(a, rows, r / "ecology/tosica_major", "model_best.pth",
                                                 r / "TOSICA_source", args.device, args.batch_size)
        subtype = np.full(a.n_obs, "", dtype=object); infos = [info]
        for par, (folder, weight) in models.items():
            ix = np.flatnonzero(major == par)
            if len(ix):
                label, _, info = tosica.predict(a, ix, r / "ecology" / folder, weight,
                                               r / "TOSICA_source", args.device, args.batch_size)
                subtype[ix] = label; infos.append(info)
        labels, _, info = tosica.predict(a, rows, r / "cnv_major", "model_best.pth",
                                        r / "TOSICA_source", args.device, args.batch_size)
        cnv_major = pd.Series(labels, index=a.obs_names); infos.append(info)
    cells = pd.DataFrame({"sampleID": a.obs.sampleID.to_numpy(), "is_tumor": a.obs.is_tumor.to_numpy(),
                          "major": major, "subtype": subtype, "cnv_major": cnv_major.to_numpy()}, index=a.obs_names)
    cells.to_csv(out / "annotations.csv.gz", index_label="cell_id")
    fractions = cell_fractions(cells[cells.is_tumor], PANEL18, supported)
    expression = infer_cnv(a, cnv_major, r / "gene_positions.tsv", out)
    features = fractions.merge(expression, on="sampleID", how="left", validate="one_to_one")
    features.insert(0, "dataset", args.dataset)
    required = PANEL18 + ["weighted_MLH1"]
    valid = np.isfinite(features[required].to_numpy(dtype=float)).all(axis=1)
    features.assign(scoring_status=np.where(valid, "complete", "unmeasured_feature_abstain")).to_csv(out / "all_samples_features.csv", index=False)
    if not valid.any():
        raise ValueError("No complete fixed18+weightedMLH1 inputs; see all_samples_features.csv; no zero imputation")
    score_features(features.loc[valid].copy(), out, "scrna")
    manifest(out, {"modality": "scrna", "n_scored": int(valid.sum()), "n_abstained": int((~valid).sum()),
                   "annotation_models": infos, "reference_files": records, "model_refit": False,
                   "ecology_checkpoint_identity": "historical hybrid Major+CRC; original/unknown training seeds preserved",
                   "CNV_checkpoint_identity": "seed2026 Major; Strom reference only",
                   "interpretation": "Portable expression-input route using fixed resources; its output is a new technical run, not a claim that archived AUROC has been rerun."})
