from pathlib import Path
import json, hashlib, sys
import numpy as np, pandas as pd, h5py, anndata as ad, joblib
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
ROOT=Path('/home/wyw/rstudio/MSI'); F=ROOT/'最终模型'; B=F/'data/scRNA/result/diagnostics'; O=Path(__file__).resolve().parent
SOURCE=B/'shared18_malignant_fraction_weighted_mlh1_seed2026'; BR=B/'infercnvpy_stromal_only_unique_seed2026'
counts=pd.read_csv(SOURCE/'sample_counts_and_weights.csv'); ledger=pd.read_csv(B/'shared18_same_sample_benchmark_seed2026/sample_ledger.csv'); trainmeta=ledger[ledger.dataset.eq('GSE178341')].set_index('sampleID')
art=joblib.load(F/'model/reference/shared18_weighted_mlh1_2026/model.joblib'); features=art['features']; template=art['pipeline']
qcrows=[]; diag=[]; sens=[]; norm=[]; ihc=[]; scoretables={}; obss={}; sources=[]
def loadcsv(p,**kw):
 sources.append(str(p)); return pd.read_csv(p,**kw)
for ds in ['GSE178341','GSE132465','GSE200997']:
 p=ROOT/(ds+'.h5ad') if ds!='GSE200997' else F/'data/scRNA/external/GSE200997/GSE200997.h5ad'
 with h5py.File(p,'r') as f:
  obs=ad.io.read_elem(f['obs']); var=ad.io.read_elem(f['var']); obss[ds]=obs
  names=var.index.astype(str); j=np.flatnonzero(names=='MLH1'); assert len(j)==1
  x=ad.io.sparse_dataset(f['X']); raw=ad.io.sparse_dataset(f['layers/counts']) if isinstance(f['layers/counts'],h5py.Group) else f['layers/counts']
  idx=np.unique(np.linspace(0,len(obs)-1,128,dtype=int)); xx=x[idx,:].toarray().astype(float); rr=raw[idx,:]; cc=(rr.toarray() if hasattr(rr,'toarray') else np.asarray(rr)).astype(float)
  lib=cc.sum(1); predicted=np.log1p(cc/lib[:,None]*10000); err=np.max(np.abs(xx-predicted),axis=1)
  inferred=np.expm1(xx).sum(1)
  norm.append(dict(dataset=ds,layer='X',counts_layer='layers/counts',n_genes=len(var),audit_cells=len(idx),inferred_total_min=inferred.min(),inferred_total_median=np.median(inferred),inferred_total_max=inferred.max(),max_abs_error_log1p10k=float(err.max()),n_cells_matching_log1p10k=int((err<1e-5).sum()),history='uns has no normalization provenance; numeric audit of 128 evenly spaced cells'))
  mlh=np.zeros(len(obs))
  for start in range(0,len(obs),4096):mlh[start:start+4096]=x[start:start+4096,:][:,int(j[0])].toarray().ravel()
 sample=obs['sample' if ds=='GSE178341' else 'Sample'].astype(str)
 tumor=obs['cancer'].astype(str).eq('tumor') if ds=='GSE178341' else obs.Class.astype(str).eq('Tumor')
 if ds=='GSE178341':epi=obs.clTopLevel.astype(str).eq('Epi')
 elif ds=='GSE132465':epi=obs.Cell_type.astype(str).eq('Epithelial cells')
 else:
  mig=loadcsv(F/'data/scRNA/result/seed2026_migration_qc/GSE200997/process/GSE200997_tosica_migration_cells.csv',index_col='cell_id');epi=mig.major_prediction.reindex(obs.index).eq('Epi')
  oldmig=loadcsv(B/'GSE200997_locked_shared18/process/GSE200997_tosica_migration_cells.csv',index_col='cell_id');excluded_mask=sample.eq('T_cac5');epi.loc[excluded_mask]=oldmig.major_prediction.reindex(obs.index[excluded_mask]).eq('Epi')
 scores=loadcsv(BR/ds/'cell_malignancy_scores.csv.gz',index_col=0); assert scores.index.is_unique
 scoretables[ds]=scores
 maskref=scores.cnv_reference.fillna('').ne(''); ref=scores.loc[maskref,'cnv_score']; q95=ref.quantile(.95)
 scored_epi=scores.Class.eq('Tumor') & epi.reindex(scores.index) & ~maskref
 c=counts[counts.dataset.eq(ds)].set_index('sampleID'); total=sample[tumor].value_counts(); full_epi=sample[tumor & epi].value_counts()
 for sid,n in total.sort_index().items():
  row=c.loc[sid] if sid in c.index else None
  keep=sid in c.index
  qcrows.append(dict(dataset=ds,sampleID=sid,total_cells_before_CNV=n,epithelial_before_CNV=int(full_epi.get(sid,0)),total_cells_retained_CNV=int((scores.Sample.eq(sid)&scores.Class.eq('Tumor')).sum()),epithelial_retained_CNV=int(row.n_epithelial) if keep else np.nan,candidate_malignant_q95=int(row.n_malignant) if keep else np.nan,p=float(row.applied_weight) if keep else np.nan,mu=float(row.raw_MLH1) if keep else np.nan,W=float(row.weighted_MLH1) if keep else np.nan,qc_pass=n>=200,included=keep,epithelial_annotation=('historical pre-QC TOSICA only; no current annotation' if sid=='T_cac5' else ('current seed2026_migration_qc TOSICA' if ds=='GSE200997' else 'input h5ad epithelial annotation')),reason='' if keep else ('total cells <200; no current CNV/weighted score' if n<200 else 'not in current labelled analysis set'),measurement='computable; few cells (<10), descriptive flag only' if keep and row.n_malignant<10 else ('computable; precision not established' if keep else 'not scored'),candidate_before_sampling='not identified on unsampled cells' if ds=='GSE178341' else ('same retained set' if keep else 'not available')))
 for q in [.90,.95,.99]:
  t=ref.quantile(q); candidate=scored_epi & scores.cnv_score.ge(t-1e-12)
  strict=scored_epi & scores.cnv_score.gt(t); inclusive=scored_epi & scores.cnv_score.ge(t)
  diag.append(dict(dataset=ds,quantile=q,n_reference=len(ref),n_unique_reference_scores=ref.nunique(),ref_min=ref.min(),ref_median=ref.median(),ref_max=ref.max(),threshold=t,n_ref_at_threshold=int(ref.eq(t).sum()),n_ref_within_tolerance=int((ref-t).abs().le(1e-12).sum()),n_epithelial=int(scored_epi.sum()),n_candidates=int(candidate.sum()),additional_vs_strict_gt=int((candidate&~strict).sum()),additional_from_tolerance=int((candidate&~inclusive).sum())))
  selected=scores[candidate].copy();selected['MLH1']=pd.Series(mlh,index=obs.index).reindex(selected.index)
  summ=selected.groupby('Sample').agg(n=('MLH1','size'),mu=('MLH1','mean'))
  for sid,row in c.iterrows():
   n=int(summ.loc[sid,'n']) if sid in summ.index else 0; mu=float(summ.loc[sid,'mu']) if n else np.nan; w=mu*n/row.n_epithelial if n else np.nan
   if q==.95:
    assert n==row.n_malignant;np.testing.assert_allclose([mu,w],[row.raw_MLH1,row.weighted_MLH1],rtol=1e-6,atol=1e-9)
   sens.append(dict(dataset=ds,sampleID=sid,quantile=q,threshold=t,n_epithelial=int(row.n_epithelial),n_candidate=n,p=n/row.n_epithelial,mu=mu,W=w,scoreable=n>0))
 print('Audited',ds,flush=True)
pd.DataFrame(qcrows).to_csv(O/'S1_sample_QC_and_counts.csv',index=False)
pd.DataFrame(diag).to_csv(O/'S2_CNV_reference_diagnostics.csv',index=False)
pd.DataFrame(sens).to_csv(O/'S3_q90_q95_q99_measurement_sensitivity.csv',index=False)
pd.DataFrame(norm).to_csv(O/'S4_expression_layer_audit.csv',index=False)
# Raw source IHC versus stored annotations and actual model endpoint.
meta=loadcsv(F/'data/scRNA/reference/metadata_sources/GSE178341_crc10x_full_c295v4_submit_metatables.csv.gz',usecols=['cellID','PatientTypeID','PID','MMR_IHC','MMRStatus','MLH1Status'])
obs=obss['GSE178341']
for sid in ['C106T','C115T','C132T']:
 local=obs[obs['sample'].astype(str).eq(sid)];matched=meta[meta.cellID.isin(local.index)]
 assert len(matched)==len(local)
 ihc.append(dict(sampleID=sid,source_patient_type=';'.join(matched.PatientTypeID.unique()),matched_cell_ids=len(matched),source_MMR_IHC=';'.join(matched.MMR_IHC.dropna().unique()),source_MMRStatus=';'.join(matched.MMRStatus.dropna().unique()),source_MLH1_methylation=';'.join(matched.MLH1Status.dropna().unique()),local_MLH1=';'.join(local.MLH1.astype(str).unique()),model_MSI_endpoint=trainmeta.loc[sid,'MMR'],conclusion='cell-ID mapping agrees; local express/loss provenance unresolved; methylation is distinct from IHC; do not infer RNA-protein discordance'))
pd.DataFrame(ihc).to_csv(O/'S5_IHC_traceability.csv',index=False)
# Audit archived paired predictions; do not refit the existing four models.
pred=loadcsv(F/'report/article/weighted_scRNA_update/all_model_predictions.csv')
metrics=[]
for key,g in pred.groupby(['dataset','split','model']): metrics.append(dict(zip(['dataset','split','model'],key),n=len(g),AUROC=roc_auc_score(g.y,g.score)))
pd.DataFrame(metrics).to_csv(O/'existing_AUROC_verified.csv',index=False)
pairs=loadcsv(F/'report/article/weighted_scRNA_update/paired_ablation_AUROC_differences.csv');pairs.to_csv(O/'S6_existing_paired_AUROC_differences.csv',index=False)
# Additional controls: same frozen panel, patient folds and classifier template; no selection.
tables={ds:loadcsv(SOURCE/f'{ds}_raw_MLH1_refit_features.csv',index_col=0) for ds in counts.dataset.unique()}
for ds,t in tables.items():t['p']=counts[counts.dataset.eq(ds)].set_index('sampleID').applied_weight.loc[t.index]
y=counts[counts.dataset.eq('GSE178341')].set_index('sampleID').MMR.eq('MSI').astype(int)
train=tables['GSE178341']; folds=trainmeta.loc[train.index,'fold'];newpred=[]
for name,cols in [('p only',['p']),('18 subtypes + p',features[:-1]+['p']),('18 subtypes + mu + p',features+['p'])]:
 for fold in sorted(folds.unique()):
  test=folds.eq(fold);assert not set(trainmeta.loc[train.index[test],'patientID'])&set(trainmeta.loc[train.index[~test],'patientID'])
  model=clone(template).fit(train.loc[~test,cols],y.loc[train.index[~test]])
  for sid,s in zip(train.index[test],model.predict_proba(train.loc[test,cols])[:,1]):newpred.append(dict(dataset='GSE178341',split='grouped_OOF',model=name,sampleID=sid,y=int(y.loc[sid]),score=s))
 model=clone(template).fit(train[cols],y.loc[train.index])
 for ds,t in tables.items():
  if ds=='GSE178341':continue
  labels=counts[counts.dataset.eq(ds)].set_index('sampleID').MMR.eq('MSI').astype(int)
  for sid,s in zip(t.index,model.predict_proba(t[cols])[:,1]):newpred.append(dict(dataset=ds,split='external',model=name,sampleID=sid,y=int(labels.loc[sid]),score=s))
newpred=pd.DataFrame(newpred);newpred.to_csv(O/'additional_control_predictions.csv',index=False)
rng=np.random.default_rng(2026);out=[];newpairs=[]
for ds,split in [('GSE178341','grouped_OOF'),('GSE132465','external'),('GSE200997','external')]:
 g=pd.concat([pred,newpred],ignore_index=True);g=g[g.dataset.eq(ds)&g.split.eq(split)]
 wide=g.pivot(index='sampleID',columns='model',values='score'); labels=g.drop_duplicates('sampleID').set_index('sampleID').y.loc[wide.index]
 patients=trainmeta.loc[wide.index,'patientID'] if ds=='GSE178341' else pd.Series(wide.index,index=wide.index)
 fr=pd.DataFrame(dict(patient=patients,y=labels));assert fr.groupby('patient').y.nunique().eq(1).all()
 groups=[fr.loc[fr.y.eq(k),'patient'].unique() for k in [0,1]]; imap={pid:np.flatnonzero(patients.to_numpy()==pid) for pid in patients.unique()}
 # Rank-pair AUROC, identical to roc_auc_score, much cheaper for bootstrap.
 allscores=wide.to_numpy();a=[]
 for _ in range(2000):
  ids=[np.concatenate([imap[pid] for pid in rng.choice(gp,len(gp),replace=True)]) for gp in groups]
  pos=allscores[ids[1]][:,None,:];neg=allscores[ids[0]][None,:,:]
  a.append(((pos>neg)+.5*(pos==neg)).mean(axis=(0,1)))
 a=np.array(a); observed=np.array([roc_auc_score(labels,wide[n]) for n in wide.columns]);joint=list(wide.columns).index('18 subtypes + weighted MLH1')
 for j,name in enumerate(wide.columns):
  lo,hi=np.quantile(a[:,j],[.025,.975]);out.append(dict(dataset=ds,split=split,model=name,n=len(wide),AUROC=observed[j],CI_low=lo,CI_high=hi))
  if name in newpred.model.unique():
   lo,hi=np.quantile(a[:,joint]-a[:,j],[.025,.975]);newpairs.append(dict(dataset=ds,baseline=name,delta=observed[joint]-observed[j],CI_low=lo,CI_high=hi))
pd.DataFrame(out).to_csv(O/'S7_extended_controls.csv',index=False);pd.DataFrame(newpairs).to_csv(O/'extended_controls_paired_differences.csv',index=False)
# Same archived model scores, 16 versus 15 samples: only historical proxy, not current weighted model.
old=loadcsv(B/'GSE200997_locked_shared18/local_predictions.csv');q=loadcsv(B/'GSE200997_qc_total_cells_gt200/sample_cell_QC_decision.csv');keep=set(q[q.qc_pass].sampleID);inc=[]
for name,g in old.groupby('model'):
 for subset,part in [('all16',g),('QC15',g[g.sampleID.isin(keep)])]:
  inc.append(dict(model=name,subset=subset,n=len(part),AUROC=roc_auc_score(part.y,part.score),identity='historical archived model; unchanged predictions; not current weighted classifier'))
pd.DataFrame(inc).to_csv(O/'S8_historical_same_model_QC_comparison.csv',index=False)
# Tiny-cell omission is a sensitivity display only, never a new eligibility rule.
low=counts[counts.n_malignant.lt(10)][['dataset','sampleID']];om=[]
for ds,split in [('GSE178341','grouped_OOF'),('GSE132465','external'),('GSE200997','external')]:
 ids=set(low[low.dataset.eq(ds)].sampleID)
 for name,g in pred[pred.dataset.eq(ds)&pred.split.eq(split)].groupby('model'):
  part=g[~g.sampleID.isin(ids)];om.append(dict(dataset=ds,model=name,omitted=';'.join(sorted(ids)),n=len(part),AUROC=roc_auc_score(part.y,part.score),interpretation='descriptive subset of unchanged predictions, no retraining; <10 is not eligibility criterion'))
pd.DataFrame(om).to_csv(O/'S9_low_cell_count_sensitivity.csv',index=False)
# Preserve actual source hashes and timestamp evidence separately from scientific prespecification.
manifest=dict(seed=2026,scope='manuscript evidence audit plus fixed additional controls; no model selection, no replacement of active model',source_hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sorted(set(sources))},pending=['full nested weighted development evaluation','new unexposed cohort evaluation','bulk 4+4 reanalysis','CNV reclustering resolution sensitivity','repeated subsampling and CNV reruns','original normalization creation logs','dated QC threshold decision record'])
(O/'evidence_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print(pd.DataFrame(out).to_string(index=False));print(pd.DataFrame(norm).to_string(index=False));print(pd.DataFrame(ihc).to_string(index=False))
