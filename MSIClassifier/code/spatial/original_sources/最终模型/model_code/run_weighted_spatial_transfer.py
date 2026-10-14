"""Rebuild spatial reference with CNV-defined epithelial states; fixed-model transfer."""
from pathlib import Path
import sys
import json
import random
import gc
import time
import hashlib

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import torch
import scvi
import joblib
from lightning.pytorch.callbacks import Callback
from cell2location.models import Cell2location, RegressionModel
from cell2location.utils.filtering import filter_genes
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
FINAL=ROOT/'最终模型'
BASE=FINAL/'data/scRNA/result/diagnostics'
OUT=FINAL/'data/stRNA/result/weighted_mlh1_transfer_seed2026'
sys.path.insert(0,str(ROOT/'stRNA_result'))
import run_cell2location_gse178341_spatial as old

SEED=2026
MAL='Epi_CNV_candidate'
OTHER='Epi_other'
CNV=BASE/'infercnvpy_stromal_only_unique_seed2026/GSE178341/cell_malignancy_scores.csv.gz'
CELLS=BASE/'infercnvpy_stromal_mlh1_impact_q95_training_unique_seed2026/GSE178341_selected_epithelial_cells.csv.gz'
CONTRACT=BASE/'unmerged_raw_subtypes_min50_20260908/class_contract_71_unmerged.csv'
CLASSIFIER=FINAL/'model/reference/shared18_weighted_mlh1_2026/model.joblib'

class TrainingProgress(Callback):
    def __init__(self,label):
        self.label=label
    def on_train_epoch_end(self,trainer,pl_module):
        if (trainer.current_epoch+1)%100==0:
            values={k:float(v.detach().cpu()) for k,v in trainer.callback_metrics.items() if torch.is_tensor(v) and v.numel()==1}
            print(f'EPOCH {self.label} {trainer.current_epoch+1} {values}',flush=True)

def seed():
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED);scvi.settings.seed=SEED
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.set_num_threads(4)

def reference(samples):
    refdir=OUT/'reference';refdir.mkdir(exist_ok=True)
    if (refdir/'complete.json').exists():
        return pd.read_csv(refdir/'signature.csv',index_col=0),pd.read_csv(refdir/'mapping.csv')
    seed()
    original=ad.read_h5ad(ROOT/'GSE178341.h5ad',backed='r')
    obs=original.obs.copy();assert obs.index.is_unique
    contract=pd.read_csv(CONTRACT).set_index('class_name')
    cnv=pd.read_csv(CNV,index_col=0);assert cnv.index.is_unique
    malignant=pd.read_csv(CELLS,index_col=0).index
    assert malignant.is_unique
    labels=pd.Series('',index=obs.index)
    tumor=obs.cancer.astype(str).eq('tumor')
    non=obs.cl295v11SubShort.astype(str).isin(contract.index)&tumor&obs.clTopLevel.astype(str).ne('Epi')
    labels.loc[non]=obs.loc[non,'cl295v11SubShort'].astype(str)
    epi=tumor&obs.clTopLevel.astype(str).eq('Epi')&obs.index.isin(cnv.index)
    labels.loc[epi]=OTHER;labels.loc[epi&obs.index.isin(malignant)]=MAL
    assert labels.eq(MAL).sum()==len(malignant)
    counts=labels[labels.ne('')].value_counts().sort_index()
    assert len(counts)==73 and counts.ge(50).all()
    rng=np.random.default_rng(SEED);ids=[]
    for label in counts.index:
        candidates=np.flatnonzero(labels.eq(label))
        ids.extend(rng.choice(candidates,min(1200,len(candidates)),replace=False))
    ids=np.sort(ids)
    pool=old.get_spatial_gene_symbols(samples[0])&old.read_common_genes(ROOT/'bulk_data/standardized_expression_common_genes.txt')
    genes=original.var_names.isin(pool)
    assert 'counts' in original.layers
    # Only retain count data and the annotations needed for the reference model.
    selected=original[ids,genes].to_memory();original.file.close()
    x=selected.layers['counts'].copy()
    ref=ad.AnnData(X=x,obs=pd.DataFrame(index=selected.obs_names),var=pd.DataFrame(index=selected.var_names))
    ref.obs['small_type']=labels.loc[ref.obs_names].astype('category')
    ref.obs['reference_sample']=obs.loc[ref.obs_names,'sample'].astype(str).astype('category')
    del selected,x;gc.collect()
    ref=old.collapse_duplicate_genes(ref);ref.X=ref.X.astype(np.float32)
    ref.obs[['small_type','reference_sample']].to_csv(refdir/'selected_reference_cells.csv.gz',compression='gzip')
    counts.rename('n_eligible').to_csv(refdir/'eligible_reference_counts.csv')
    mapping=pd.DataFrame({'small_type':counts.index,'major_type':[contract.loc[t,'group_class'] if t in contract.index else 'Epi' for t in counts.index]})
    mapping.to_csv(refdir/'mapping.csv',index=False)
    keep=filter_genes(ref,cell_count_cutoff=15,cell_percentage_cutoff2=.03,nonz_mean_cutoff=1.12)
    plt.savefig(refdir/'gene_filter.png');plt.close('all')
    ref=ref[:,keep].copy()
    assert 'MLH1' in ref.var_names, 'MLH1 did not pass fixed reference gene filter'
    RegressionModel.setup_anndata(ref,labels_key='small_type',batch_key='reference_sample')
    model=RegressionModel(ref)
    print(f'REFERENCE_START cells={ref.n_obs} genes={ref.n_vars}',flush=True)
    old.run_train(model,dict(max_epochs=180,batch_size=2500,train_size=1,lr=.002,accelerator='gpu',enable_progress_bar=False,logger=None))
    ref=old.export_posterior(model,ref,200,2500,True)
    signature=ref.varm['means_per_cluster_mu_fg'].copy()
    signature.columns=[str(c).replace('means_per_cluster_mu_fg_','') for c in signature.columns]
    assert np.isfinite(signature.to_numpy()).all() and (signature>0).all().all()
    signature.to_csv(refdir/'signature.csv');model.save(str(refdir/'model'),overwrite=True,save_anndata=False)
    for key,val in model.history.items():val.to_csv(refdir/f'history_{key}.csv')
    (refdir/'complete.json').write_text(json.dumps(dict(n_cells=ref.n_obs,n_genes=ref.n_vars,n_types=73,seed=SEED,
        reference_epochs=180,posterior_samples=200,reference='tumor non-epithelial71 plus q95 candidate and other scored tumor epithelium',
        no_normal_claim=True,source_cnv=str(CNV)),indent=2))
    del ref,model;gc.collect();torch.cuda.empty_cache()
    print('REFERENCE_DONE',flush=True)
    return signature,mapping

def malignant_counts(model,adata):
    # Exact official conditional-expression formula, restricted to one type to avoid
    # materializing 73 full gene-by-spot matrices. Verify against official API below.
    samples=model.samples['post_sample_means'];module=model.module.model
    factors=list(model.factor_names_);idx=factors.index(MAL)
    matrix=module.cell_state_mat
    if torch.is_tensor(matrix):matrix=matrix.detach().cpu().numpy()
    mu=samples['w_sf']@matrix.T*samples['m_g']+samples['s_g_gene_add'][0]
    component=samples['w_sf'][:,idx,None]*matrix[:,idx][None,:]*samples['m_g']
    assert np.isfinite(mu).all() and (mu>0).all()
    result=sp.csr_matrix(adata.X).multiply(component/mu).tocsr()
    official=module.compute_expected_per_cell_type(samples,model.adata_manager,ind_x=np.arange(min(8,adata.n_obs)))['mu'][idx]
    np.testing.assert_allclose(result[:official.shape[0]].toarray(),official.toarray(),atol=1e-5,rtol=1e-5)
    assert np.all(result.data>=0)
    return result

def spatial_one(folder,signature,mapping):
    output=OUT/folder.name;output.mkdir(exist_ok=True)
    if (output/'complete.json').exists():return
    seed();started=time.time()
    a=old.load_spatial_sample(folder,signature.index)
    assert 'MLH1' in a.var_names
    Cell2location.setup_anndata(a)
    model=Cell2location(a,cell_state_df=signature.loc[a.var_names],N_cells_per_location=8,detection_alpha=20)
    print(f'SPATIAL_START {folder.name} spots={a.n_obs} genes={a.n_vars}',flush=True)
    old.run_train(model,dict(max_epochs=500,batch_size=min(1024,a.n_obs),train_size=1,lr=.002,accelerator='gpu',enable_progress_bar=False,logger=None,callbacks=[TrainingProgress(folder.name)]))
    a=old.export_posterior(model,a,200,min(1024,a.n_obs),True)
    abundance=old.clean_abundance_columns(a.obsm['means_cell_abundance_w_sf'],'meanscell_abundance_w_sf_')
    assert set(abundance.columns)==set(signature.columns)
    abundance.to_csv(output/'small_cell_abundance_means.csv')
    reconstructed=malignant_counts(model,a)
    malignant_total=np.asarray(reconstructed.sum(axis=1)).ravel()
    gene=np.asarray(reconstructed[:,a.var_names.get_loc('MLH1')].toarray()).ravel()
    value=np.log1p(np.divide(gene*10000,malignant_total,out=np.zeros_like(gene),where=malignant_total>0))
    ma=abundance[MAL].to_numpy();other=abundance[OTHER].to_numpy();epi=ma+other
    assert epi.sum()>0 and ma.sum()>0 and (malignant_total>0).all()
    mean=float(np.sum(value*ma)/ma.sum()); fraction=float(ma.sum()/epi.sum());weighted=mean*fraction
    spot=pd.DataFrame({'candidate_abundance':ma,'other_epithelial_abundance':other,
        'candidate_fraction':ma/epi,'candidate_MLH1_reconstructed_log1p_cp10k':value,
        'candidate_MLH1_allocated_counts':gene,'candidate_library_allocated_counts':malignant_total,
        'x':a.obsm['spatial'][:,0],'y':a.obsm['spatial'][:,1]},index=a.obs_names)
    spot.to_csv(output/'spot_mlh1_and_abundance.csv.gz',compression='gzip')
    features={}
    classifier=joblib.load(CLASSIFIER)
    groups=mapping.set_index('small_type').major_type
    sums=abundance.sum(axis=0)
    for feature in classifier['features'][:-1]:
        parent=groups.loc[feature];denom=sums.loc[groups[groups==parent].index].sum()
        assert denom>0
        features[feature]=float(sums[feature]/denom)
    features['mean_MLH1_epithelial']=weighted
    pd.DataFrame([features],index=[folder.name.split('(')[0]]).to_csv(output/'classifier_features.csv',index_label='sampleID')
    model.save(str(output/'cell2location_model'),overwrite=True,save_anndata=False)
    np.savez_compressed(output/'posterior_means.npz',**model.samples['post_sample_means'])
    for key,val in model.history.items():val.to_csv(output/f'history_{key}.csv')
    assert model.history, 'Spatial training history was not recorded'
    (output/'complete.json').write_text(json.dumps(dict(sample=folder.name,n_spots=a.n_obs,n_genes=a.n_vars,
        n_types=len(abundance.columns),seed=SEED,spatial_epochs=500,posterior_samples=200,
        candidate_MLH1_mean_proxy=mean,candidate_fraction=fraction,weighted_MLH1_proxy=weighted,
        expression='official conditional count allocation; malignant spot profile normalized to 10000 counts and log1p; candidate-abundance weighted across spots',
        limitation='reconstructed malignant spot profile is not an observed mean of single-cell log expression; shared-gene normalization and platform scales may differ',
        elapsed_seconds=time.time()-started,official_expression_formula_verified=True),indent=2))
    print(f'SPATIAL_DONE {folder.name} seconds={time.time()-started:.1f} weighted_MLH1={weighted:.6g}',flush=True)
    del model,a,reconstructed;gc.collect();torch.cuda.empty_cache()

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker-index',type=int,default=0)
    parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args()
    assert args.workers>=1 and 0<=args.worker_index<args.workers
    OUT.mkdir(parents=True,exist_ok=True)
    samples=old.list_spatial_samples(ROOT/'空间转录组')
    assert len(samples)==14
    (OUT/'analysis_plan.json').write_text(json.dumps(dict(seed=SEED,reference_epochs=180,spatial_epochs=500,posterior_samples=200,
        reference_types=73,N_cells_per_location=8,detection_alpha=20,external_labels_used_for_fitting=False,
        fixed_classifier=str(CLASSIFIER),classifier_sha256=hashlib.sha256(CLASSIFIER.read_bytes()).hexdigest(),
        formula='weighted_MLH1_proxy = sum_spots(reconstructed_malignant_MLH1 * malignant_abundance) / sum_spots(all_epithelial_abundance)',
        interpretation='exploratory cross-platform feature reconstruction; not exact measurement equivalence',
        historical_spatial_hyperparameters='180 reference epochs, 500 spatial epochs, 200 posterior samples; convergence must be audited'),indent=2))
    signature,mapping=reference(samples)
    for folder in samples[args.worker_index::args.workers]:spatial_one(folder,signature,mapping)
    print(f'WORKER_COMPLETE {args.worker_index}/{args.workers}',flush=True)

if __name__=='__main__':main()
