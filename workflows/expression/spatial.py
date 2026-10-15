"""One raw section -> current 73-type mapping -> reconstructed proxy -> frozen LDA."""
import random
import numpy as np
import pandas as pd
from scipy import sparse
from inputs import PANEL17, parent, spatial_weighted_mlh1
from runners import require_resources, score_features, manifest


def run(args, a, out):
    import torch
    import scvi
    from cell2location.models import Cell2location
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; select --device cpu")
    records = require_resources(args.resource_dir, ["spatial/"])
    random.seed(2026); np.random.seed(2026); torch.manual_seed(2026)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2026)
    scvi.settings.seed = 2026
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    signature = pd.read_csv(args.resource_dir / "spatial/signature.csv", index_col=0)
    mapping = pd.read_csv(args.resource_dir / "spatial/mapping.csv").set_index("small_type").major_type
    if signature.shape[1] != 73 or set(signature.columns) != set(mapping.index):
        raise ValueError("Use the fixed current73-type signature and matching parent map")
    genes = signature.index.intersection(a.var_names)
    if len(genes) < 2000 or "MLH1" not in genes:
        raise ValueError("Require MLH1 and at least2000 shared signature genes")
    a = a[:, genes].copy()
    Cell2location.setup_anndata(a)
    model = Cell2location(a, cell_state_df=signature.loc[genes], N_cells_per_location=8, detection_alpha=20)
    accelerator = "gpu" if args.device.startswith("cuda") else "cpu"
    model.train(max_epochs=500, batch_size=min(1024, a.n_obs), train_size=1, lr=.002,
                accelerator=accelerator, enable_progress_bar=False, logger=None)
    a = model.export_posterior(a, sample_kwargs={"num_samples": 200, "batch_size": min(1024, a.n_obs), "accelerator": accelerator})
    abundance = a.obsm["means_cell_abundance_w_sf"].copy()
    abundance.columns = [str(c).replace("meanscell_abundance_w_sf_", "") for c in abundance]
    samples = model.samples["post_sample_means"]
    cell_state = model.module.model.cell_state_mat
    if torch.is_tensor(cell_state):
        cell_state = cell_state.detach().cpu().numpy()
    idx = list(model.factor_names_).index("Epi_CNV_candidate")
    mu = samples["w_sf"] @ cell_state.T * samples["m_g"] + samples["s_g_gene_add"][0]
    component = samples["w_sf"][:, idx, None] * cell_state[:, idx][None, :] * samples["m_g"]
    if not np.isfinite(mu).all() or (mu <= 0).any():
        raise ValueError("Invalid posterior expected-expression denominator")
    allocated = sparse.csr_matrix(a.X).multiply(component / mu).tocsr()
    official = model.module.model.compute_expected_per_cell_type(samples, model.adata_manager,
               ind_x=np.arange(min(8, a.n_obs)))["mu"][idx]
    np.testing.assert_allclose(allocated[:official.shape[0]].toarray(), official.toarray(), atol=1e-5, rtol=1e-5)
    proxy = spatial_weighted_mlh1(allocated, a.var_names.get_loc("MLH1"), abundance)
    row = {"sampleID": args.sample_id}
    totals = abundance.sum(axis=0)
    for feature in PANEL17:
        denominator = totals.loc[mapping[mapping.eq(parent(feature))].index].sum()
        if denominator <= 0:
            raise ValueError("Missing spatial parent denominator for " + feature)
        row[feature] = float(totals[feature] / denominator)
    row["weighted_MLH1"] = proxy["weighted_MLH1"]
    abundance.to_csv(out / "abundance.csv", index_label="spot_id")
    pd.DataFrame({"candidate_MLH1_log1p_cp10k": proxy["spot_MLH1"], "candidate_allocated_MLH1": proxy["allocated_MLH1"],
                  "candidate_allocated_library": proxy["allocated_library"]}, index=a.obs_names).to_csv(out / "spot_proxy.csv.gz", index_label="spot_id")
    np.savez_compressed(out / "posterior_means.npz", **samples)
    model.save(str(out / "cell2location_model"), overwrite=False, save_anndata=True)
    for key, values in model.history.items():
        values.to_csv(out / ("history_" + key + ".csv"))
    if not model.history:
        raise ValueError("Spatial training history was not retained")
    score_features(pd.DataFrame([row]), out, "spatial")
    manifest(out, {"modality": "spatial", "sample_id": args.sample_id, "n_spots": a.n_obs, "n_genes": len(genes),
                   "reference_files": records, "reference_refitted": False, "spatial_epochs": 500,
                   "posterior_draws": 200, "N_cells_per_location": 8, "detection_alpha": 20,
                   "reference_identity": "saved current73-type signature from180-epoch regression; reference epoch history absent",
                   "formula": "sum_spots(candidate log1p CP10k MLH1 * candidate abundance) / sum_spots(all epithelial abundance)",
                   "normalization_genes": "all shared retained signature genes",
                   "interpretation": "Exploratory reconstructed spatial proxy; no measurement-equivalence or convergence claim."})
