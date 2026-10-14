"""Read saved scRNA evidence and replay metadata/scores only; never refit models."""
from pathlib import Path
import hashlib,json
from scipy.stats import spearmanr
import pandas as pd,numpy as np,h5py,anndata as ad,joblib
from sklearn.metrics import roc_auc_score

ROOT=Path('/home/wyw/rstudio/MSI')
FINAL=ROOT/'最终模型'
ARTICLE=FINAL/'report/article'
OUT=Path(__file__).resolve().parent
SUPP=ARTICLE/'bulk12_loco_manuscript_seed2026_20261013/supplement'
LEDGER=FINAL/'data/scRNA/result/diagnostics/shared18_same_sample_benchmark_seed2026/sample_ledger.csv'
PRED=ARTICLE/'weighted_scRNA_update/all_model_predictions.csv'
CONTROLS=ARTICLE/'revision_review_20260928/additional_control_predictions.csv'
ledger=pd.read_csv(LEDGER)
pred=pd.read_csv(PRED)
g=ledger[ledger.dataset.eq('GSE178341')]
assert len(g)==64 and g.patientID.nunique()==62
assert g.groupby('patientID').fold.nunique().eq(1).all()
q=pred[(pred.dataset.eq('GSE178341'))&(pred.split.eq('grouped_OOF'))&(pred.model.eq('18 subtypes + weighted MLH1'))].merge(g[['sampleID','patientID']],on='sampleID',validate='one_to_one')
patient=q.groupby('patientID').agg(y=('y','first'),score=('score','mean'))
checks={'no_refitting_performed':True,'sample_auc':float(roc_auc_score(q.y,q.score)),
        'audit_only_mean_patient_auc':float(roc_auc_score(patient.y,patient.score)),
        'sample_count':len(q),'patient_count':len(patient),
        'patient_positive_negative':[int(patient.y.sum()),int((patient.y==0).sum())],
        'duplicate_patient_rows':g[g.patientID.duplicated(keep=False)][['sampleID','patientID','fold','y']].to_dict('records'),
        'fold_counts':g.groupby('fold').agg(samples=('sampleID','size'),patients=('patientID','nunique')).reset_index().to_dict('records')}

book=SUPP/'SupplementaryTablesS1-S14_English.xlsx'
s7_csv=pd.read_csv(SUPP/'S7.csv')
s7_xlsx=pd.read_excel(book,sheet_name='S7')
pd.testing.assert_frame_equal(s7_csv,s7_xlsx,check_exact=False,rtol=1e-12,atol=1e-12)
allpred=pd.concat([pred,pd.read_csv(CONTROLS)],ignore_index=True)
verified=[]
for row in s7_csv.itertuples():
    a=allpred[(allpred.dataset.eq(row.dataset))&(allpred.split.eq(row.split))&(allpred.model.eq(row.model))]
    assert a.sampleID.is_unique and len(a)==row.n
    value=float(roc_auc_score(a.y,a.score))
    assert np.isclose(value,row.AUROC,atol=1e-12,rtol=1e-12)
    excel_row=int(s7_csv.index[(s7_csv.dataset.eq(row.dataset))&(s7_csv.split.eq(row.split))&(s7_csv.model.eq(row.model))][0])+2
    verified.append({'dataset':row.dataset,'split':row.split,'model':row.model,'n':row.n,'positive':int(a.y.sum()),'AUROC':value,'S7_workbook_AUROC_cell':f'E{excel_row}','S7_csv_line':excel_row,'CI_low':row.CI_low,'CI_high':row.CI_high})
checks['S7_workbook_csv_and_score_replay_match']=True
checks['S7']=verified

p_csv=pd.read_csv(SUPP/'S6_Bootstrap_p.csv')
p_xlsx=pd.read_excel(book,sheet_name='S6_bootstrap')
pd.testing.assert_frame_equal(p_csv,p_xlsx,check_exact=False,rtol=1e-12,atol=1e-12)
npz=np.load(SUPP/'paired_patient_bootstrap_indices.npz',allow_pickle=False)
rng=np.random.default_rng(2026)
bootstrap=[]
for ds,split in [('GSE178341','grouped_OOF'),('GSE132465','external')]:
    subset=pred[(pred.dataset.eq(ds))&(pred.split.eq(split))]
    scores=subset.pivot(index='sampleID',columns='model',values='score').sort_index()
    y=subset.drop_duplicates('sampleID').set_index('sampleID').y.reindex(scores.index).to_numpy()
    patient_ids=ledger[ledger.dataset.eq(ds)].set_index('sampleID').patientID.reindex(scores.index).to_numpy()
    ids=pd.Index(npz[ds+'_patients'])
    encoding=ids.get_indexer(patient_ids)
    assert (encoding>=0).all()
    py=np.array([np.unique(y[encoding==i]).item() for i in range(len(ids))])
    pids=np.flatnonzero(py==1);nids=np.flatnonzero(py==0)
    fresh=np.array([np.concatenate([rng.choice(pids,len(pids),replace=True),rng.choice(nids,len(nids),replace=True)]) for _ in range(2000)])
    assert np.array_equal(fresh,npz[ds])
    sc=np.vstack([scores['18 subtypes + weighted MLH1'],scores['18 subtypes + raw MLH1']])
    pos=sc[:,y==1];neg=sc[:,y==0]
    kernel=(pos[:,:,None]>neg[:,None,:]).astype(float)+.5*(pos[:,:,None]==neg[:,None,:])
    delta=float(roc_auc_score(y,sc[0])-roc_auc_score(y,sc[1]))
    reps=[];counts=[]
    for draw in npz[ds]:
        cnt=np.bincount(draw,minlength=len(ids))[encoding]
        pw,nw=cnt[y==1],cnt[y==0]
        reps.append(float(pw@(kernel[0]-kernel[1])@nw/(pw.sum()*nw.sum())))
        counts.append(int(cnt.sum()))
    reps=np.array(reps)
    assert np.array_equal(reps,npz[ds+'_delta'])
    p=float((1+np.count_nonzero(np.abs(reps-delta)>=abs(delta)-1e-12))/(len(reps)+1))
    record=p_csv[p_csv.dataset.eq(ds)].iloc[0]
    assert np.isclose(p,record.paired_patient_bootstrap_p_two_sided,atol=1e-12)
    bootstrap.append({'dataset':ds,'patient_draw_shape':list(npz[ds].shape),'all_saved_draws_match_one_advancing_seed2026_stream':True,'all_saved_deltas_exactly_reproduced':True,'sample_count_min':min(counts),'sample_count_max':max(counts),'delta':delta,'p_two_sided_null_centered':p})
checks['paired_bootstrap']=bootstrap
checks['GSE200997_p_test_replicates']=int(p_csv[p_csv.dataset.eq('GSE200997')].iloc[0].bootstrap_replicates)

with h5py.File(ROOT/'GSE178341.h5ad','r') as h:obs=ad.io.read_elem(h['obs'])
obs['Sample']=obs['sample'].astype(str)
obs['Class']=obs['cancer'].astype(str).map({'tumor':'Tumor','normal':'Normal'})
qc=obs.groupby('Sample',sort=False).size()
obs=obs[obs.Sample.isin(qc[qc>=200].index)]
tumor=np.flatnonzero(obs.Class.eq('Tumor'));ref=np.flatnonzero(obs.clTopLevel.astype(str).eq('Strom'))
rng=np.random.default_rng(2026)
tumor_eligible=len(tumor);reference_eligible=len(ref)
if len(tumor)>100000:tumor=rng.choice(tumor,100000,replace=False)
if len(ref)>20000:ref=rng.choice(ref,20000,replace=False)
selected=obs.iloc[np.union1d(tumor,ref)]
saved=pd.read_csv(FINAL/'data/scRNA/result/diagnostics/infercnvpy_stromal_only_unique_seed2026/GSE178341/cell_malignancy_scores.csv.gz',index_col=0)
assert set(selected.index)==set(saved.index) and selected.index.is_unique and saved.index.is_unique
checks['CNV_selection']={'eligible_tumor_tissue_cells_after_QC':tumor_eligible,'initial_draw':len(tumor),'eligible_stromal_reference':reference_eligible,'unique_union':len(selected),'retained_tumor_tissue_cells':int(selected.Class.eq('Tumor').sum()),'retained_normal_tissue_cells':int(selected.Class.eq('Normal').sum()),'exact_saved_cell_ID_set_reproduced':True,'retained_tumor_major_counts':{str(k):int(v) for k,v in selected[selected.Class.eq('Tumor')].clTopLevel.value_counts().items()}}

artifact=joblib.load(FINAL/'model/reference/shared18_weighted_mlh1_2026/model.joblib')
checks['active_classifier']={'features':artifact['features'],'formula':artifact['formula'],'variant':artifact['variant'],'seed':artifact['seed'],'scaler_class':type(artifact['pipeline'][0]).__name__,'classifier_class':type(artifact['pipeline'][1]).__name__,'solver':artifact['pipeline'][1].solver,'shrinkage':artifact['pipeline'][1].shrinkage}
loss=pd.read_csv(FINAL/'data/scRNA/result/diagnostics/mmr4_loss_vs_mlh1_001_unique2026/training_metrics.csv')
checks['distinct_MLH1_loss_target_comparison']=loss.to_dict('records')
main=pd.read_csv(ARTICLE/'weighted_scRNA_update/AUROC_with_patient_bootstrap_CI.csv')
primary=main[(main.dataset.eq('GSE178341'))&(main.split.eq('grouped_OOF'))&(main.model.eq('18 subtypes + weighted MLH1'))].iloc[0]
checks['main_weighted_CI']={'low':float(primary.CI_low),'high':float(primary.CI_high)}
paths=[PRED,CONTROLS,LEDGER,book,SUPP/'S7.csv',SUPP/'S6_Bootstrap_p.csv',SUPP/'paired_patient_bootstrap_indices.npz',FINAL/'model/reference/shared18_weighted_mlh1_2026/model.joblib',FINAL/'model/reference/shared18_weighted_mlh1_2026/manifest.json',FINAL/'model_code/run_infercnvpy_malignant_reference.py',FINAL/'model_code/evaluate_inverse_weighted_mlh1.py',FINAL/'model_code/prepare_weighted_scrna_article_results.py',ARTICLE/'revision_review_20260928/audit_evidence.py',ARTICLE/'第三版_中文版意见修订/prepare_evidence.py',FINAL/'model_code/compare_mmr4_loss_mlh1_cutoff.py']
dependence=pd.read_csv(SUPP/'S7_Correlation_VIF.csv')
pd.testing.assert_frame_equal(dependence,pd.read_excel(book,sheet_name='S7_correlation'),check_exact=False,rtol=1e-12,atol=1e-12)
dep_source=ARTICLE/'第三版_中文版意见修订/weighted_feature_dependence.csv'
pd.testing.assert_frame_equal(dependence,pd.read_csv(dep_source),check_exact=False,rtol=1e-12,atol=1e-12)
count_source=ARTICLE/'revision_review_20260928/S1_sample_QC_and_counts.csv'
countdata=pd.read_csv(count_source); dep_rows=[]
for ds,z in countdata[countdata.included].groupby('dataset'):
    z=z.dropna(subset=['p','mu','W'])
    for u,v in [('W','mu'),('W','p'),('mu','p')]:
        dep_rows.append({'dataset':ds,'n':len(z),'analysis':'Spearman correlation','feature':u,'other':v,'value':float(spearmanr(z[u],z[v]).statistic)})
    for ycol in ['W','mu','p']:
        others=[c for c in ['W','mu','p'] if c!=ycol]
        design=np.column_stack([np.ones(len(z)),z[others].to_numpy()]);target=z[ycol].to_numpy();fit=design@np.linalg.lstsq(design,target,rcond=None)[0]
        dep_rows.append({'dataset':ds,'n':len(z),'analysis':'VIF with intercept (W, mu, p)','feature':ycol,'other':'+'.join(others),'value':float(np.square(target-target.mean()).sum()/np.square(target-fit).sum())})
pd.testing.assert_frame_equal(dependence,pd.DataFrame(dep_rows),check_exact=False,rtol=1e-10,atol=1e-10)
checks['VIF_and_correlations']={'current_workbook_sheet':'S7_correlation','current_csv':str(SUPP/'S7_Correlation_VIF.csv'),'source_csv':str(dep_source),'source_code':str(ARTICLE/'第三版_中文版意见修订/prepare_evidence.py')+':84-97','workbook_csv_source_and_recalculation_match':True,'rows':dependence.to_dict('records')}
paths.extend([SUPP/'S7_Correlation_VIF.csv',dep_source,count_source])
checks['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
(OUT/'verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
summary=OUT/'audit_findings.md'
s=summary.read_text(encoding='utf-8')
s=s.replace('0.9390751336898395',str(float(primary.CI_low)))
summary.write_text(s,encoding='utf-8')
print(json.dumps({'verified_S7_rows':len(verified),'all_saved_bootstrap_indices_replayed':True,'CNV_cell_ID_selection_replayed':True,'no_model_refitting':True,'main_CI_low':float(primary.CI_low),'outputs':str(OUT)},ensure_ascii=False))
