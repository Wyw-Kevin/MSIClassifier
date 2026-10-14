"""Read-only replay of saved weighted spatial artifacts; all writes stay in audit_spatial."""
from pathlib import Path
import json, hashlib, importlib.metadata as md
import numpy as np
import pandas as pd
import scipy.sparse as sp
import h5py
from threadpoolctl import threadpool_limits

ROOT=Path('/home/wyw/rstudio/MSI')
BASE=ROOT/'最终模型/data/stRNA/result/weighted_mlh1_transfer_seed2026'
OUT=ROOT/'最终模型/report/article/method_evidence_update_20261014/audit_spatial'
MODEL=ROOT/'最终模型/model/reference/shared18_weighted_mlh1_2026/model.joblib'
MAL='Epi_CNV_candidate'

def digest(p):
    with p.open('rb') as f:
        h=hashlib.sha256()
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def read_10x(p):
    with h5py.File(p,'r') as f:
        m=f['matrix']
        x=sp.csc_matrix((m['data'][:],m['indices'][:],m['indptr'][:]),shape=tuple(m['shape'][:])).T.tocsr().astype(np.float32)
        genes=pd.Index(m['features/name'].asstr()[:])
        barcodes=pd.Index(m['barcodes'].asstr()[:])
    if genes.has_duplicates:
        codes,unique=pd.factorize(genes,sort=False)
        mapper=sp.csr_matrix((np.ones(len(codes),dtype=np.float32),(np.arange(len(codes)),codes)),shape=(len(codes),len(unique)))
        x=x@mapper; genes=pd.Index(unique)
    return x,genes,barcodes

signature=pd.read_csv(BASE/'reference/signature.csv',index_col=0)
rows=[]
with threadpool_limits(limits=4):
 for d in sorted(BASE.glob('VISDS*')):
    m=json.loads((d/'complete.json').read_text())
    s=pd.read_csv(d/'spot_mlh1_and_abundance.csv.gz',index_col=0)
    a=pd.read_csv(d/'small_cell_abundance_means.csv',index_col=0)
    ft=pd.read_csv(d/'classifier_features.csv',index_col=0)
    p=np.load(d/'posterior_means.npz')
    source=next((ROOT/'空间转录组'/d.name).glob('*filtered_feature_bc_matrix.h5'))
    x,genes,barcodes=read_10x(source)
    shared=signature.index[signature.index.isin(genes)]
    x=x[:,genes.get_indexer(shared)].tocsr()
    assert barcodes.equals(s.index) and a.index.equals(s.index)
    assert len(shared)==12079 and np.array_equal(shared,signature.index)
    matrix=signature.loc[shared,a.columns].to_numpy(dtype=np.float32)
    idx=a.columns.get_loc(MAL);gidx=shared.get_loc('MLH1')
    libraries=[];mlh1=[]
    for start in range(0,x.shape[0],256):
        stop=min(start+256,x.shape[0]);w=p['w_sf'][start:stop]
        mu=w@matrix.T*p['m_g']+p['s_g_gene_add'][0]
        component=w[:,idx,None]*matrix[:,idx][None,:]*p['m_g']
        allocated=x[start:stop].multiply(component/mu).tocsr()
        libraries.append(np.asarray(allocated.sum(axis=1)).ravel())
        mlh1.append(np.asarray(allocated[:,gidx].toarray()).ravel())
    libraries=np.concatenate(libraries);mlh1=np.concatenate(mlh1)
    reconstructed=np.log1p(10000*mlh1/libraries)
    weighted=float(np.sum(reconstructed*p['w_sf'][:,idx])/np.sum(p['w_sf'][:,idx]+p['w_sf'][:,a.columns.get_loc('Epi_other')]))
    hist=pd.read_csv(d/'history_elbo_train.csv',index_col=0).iloc[:,-1]
    r=dict(sample=d.name,n_spots=len(s),n_genes=len(shared),source_matrix_sha256=digest(source),
      posterior_means_sha256=digest(d/'posterior_means.npz'),
      posterior_abundance_max_abs_error=float(np.max(np.abs(a.to_numpy()-p['w_sf']))),
      reconstructed_MLH1_counts_max_abs_error=float(np.max(np.abs(mlh1-s.candidate_MLH1_allocated_counts))),
      reconstructed_library_max_abs_error=float(np.max(np.abs(libraries-s.candidate_library_allocated_counts))),
      reconstructed_log1p_cp10k_max_abs_error=float(np.max(np.abs(reconstructed-s.candidate_MLH1_reconstructed_log1p_cp10k))),
      saved_count_log_formula_max_abs_error=float(np.max(np.abs(np.log1p(10000*s.candidate_MLH1_allocated_counts/s.candidate_library_allocated_counts)-s.candidate_MLH1_reconstructed_log1p_cp10k))),
      weighted_replay=weighted,saved_weighted=float(ft.mean_MLH1_epithelial.iloc[0]),weighted_abs_error=abs(weighted-float(ft.mean_MLH1_epithelial.iloc[0])),
      history_epochs=len(hist),history_finite=bool(np.isfinite(hist).all()),
      history_first50_mean=float(hist.iloc[:50].mean()),history_previous50_mean=float(hist.iloc[-100:-50].mean()),history_last50_mean=float(hist.iloc[-50:].mean()),
      history_last_window_relative_change=float(abs(hist.iloc[-50:].mean()-hist.iloc[-100:-50].mean())/abs(hist.iloc[-100:-50].mean())))
    np.testing.assert_allclose(mlh1,s.candidate_MLH1_allocated_counts,atol=1e-5,rtol=1e-5)
    np.testing.assert_allclose(libraries,s.candidate_library_allocated_counts,atol=.005,rtol=1e-5)
    np.testing.assert_allclose(reconstructed,s.candidate_MLH1_reconstructed_log1p_cp10k,atol=1e-6,rtol=1e-5)
    np.testing.assert_allclose(weighted,ft.mean_MLH1_epithelial.iloc[0],atol=1e-7,rtol=1e-5)
    assert len(hist)==500 and np.isfinite(hist).all()
    rows.append(r)
    print(d.name,'PASS',flush=True)

pd.DataFrame(rows).to_csv(OUT/'saved_spatial_replay.csv',index=False)
with h5py.File(ROOT/'GSE178341.h5ad','r') as f:
    counts=f['layers/counts/data'][:1000000]
    ref=dict(adata_shape=f['X'].attrs['shape'].tolist(),expression_used="layers['counts'] copied to reference X",counts_first_million_finite=bool(np.isfinite(counts).all()),counts_first_million_nonnegative=bool((counts>=0).all()),counts_first_million_integer=bool((counts==np.round(counts)).all()),counts_first20=counts[:20].tolist(),X_first20=f['X/data'][:20].tolist(),model_expression_gene_count=signature.shape[0],model_type_count=signature.shape[1])
selected=pd.read_csv(BASE/'reference/selected_reference_cells.csv.gz',index_col=0)
ref.update(selected_n_cells=len(selected),selected_n_types=int(selected.small_type.nunique()),selected_max_per_type=int(selected.small_type.value_counts().max()),selected_min_per_type=int(selected.small_type.value_counts().min()))
(OUT/'reference_layer_audit.json').write_text(json.dumps(ref,ensure_ascii=False,indent=2))
summary=dict(passed=True,scope='saved-artifact read-only reconstruction replay; no model fitting, posterior resampling or figure edits',n_sections=len(rows),n_spots=sum(r['n_spots'] for r in rows),normalization_denominator='sum candidate-allocated counts over all 12079 shared filtered cell2location model genes',normalization_target=10000,logarithm='natural log1p, after type-specific count allocation and CP10K',posterior_summary='post_sample_means, exported from 200 samples; plug-in reconstruction from parameter means',classifier_sha256=digest(MODEL),reconstruction_script_sha256=digest(ROOT/'最终模型/model_code/run_weighted_spatial_transfer.py'),signature_sha256=digest(BASE/'reference/signature.csv'),reference_checkpoint_sha256=digest(BASE/'reference/model/model.pt'),max_reconstructed_log_error=max(r['reconstructed_log1p_cp10k_max_abs_error'] for r in rows),max_weighted_error=max(r['weighted_abs_error'] for r in rows),last50_relative_change_min=min(r['history_last_window_relative_change'] for r in rows),last50_relative_change_max=max(r['history_last_window_relative_change'] for r in rows),versions={n:md.version(n) for n in ['cell2location','scvi-tools','anndata','scanpy','torch','pyro-ppl','numpy','pandas']})
(OUT/'saved_spatial_replay_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
