
# Packaging-only path resolution; archived originals retain exact source bytes.
from pathlib import Path as _MSIPackagePath
import sys as _MSIPackageSys
for _MSIPackageParent in _MSIPackagePath(__file__).resolve().parents:
    if (_MSIPackageParent / "project_paths.py").is_file():
        _MSIPackageSys.path.insert(0, str(_MSIPackageParent))
        break
from project_paths import project_root
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ARTICLE = HERE.parent
FINAL = ARTICLE.parent.parent
OLD = ARTICLE / 'revision_review_20260928'
PRED = ARTICLE / 'weighted_scRNA_update/all_model_predictions.csv'
LEDGER = FINAL / 'data/scRNA/result/diagnostics/shared18_same_sample_benchmark_seed2026/sample_ledger.csv'

def delong_p(y, a, b):
    scores = np.vstack([a,b])
    positive, negative = scores[:,y==1], scores[:,y==0]
    kernels = (positive[:,:,None] > negative[:,None,:]).astype(float)
    kernels += .5*(positive[:,:,None] == negative[:,None,:])
    aucs = kernels.mean(axis=(1,2))
    covariance = np.cov(kernels.mean(axis=2), ddof=1)/positive.shape[1]
    covariance += np.cov(kernels.mean(axis=1), ddof=1)/negative.shape[1]
    variance = float(np.array([1.,-1.])@covariance@np.array([1.,-1.]))
    for i in range(2):
        assert np.isclose(aucs[i], roc_auc_score(y,scores[i]),atol=1e-12)
    if variance <= 1e-15:
        return float(aucs[0]-aucs[1]), np.nan, variance
    return float(aucs[0]-aucs[1]), float(2*norm.sf(abs(aucs[0]-aucs[1])/np.sqrt(variance))), variance

pred = pd.read_csv(PRED)
ledger = pd.read_csv(LEDGER)
rng = np.random.default_rng(2026)
bootstrap_draws = {}
def patient_bootstrap_p(y, a, b, patients, dataset):
    scores=np.vstack([a,b])
    pos,neg=scores[:,y==1],scores[:,y==0]
    kernels=(pos[:,:,None]>neg[:,None,:]).astype(float)+.5*(pos[:,:,None]==neg[:,None,:])
    aucs=kernels.mean(axis=(1,2))
    for i in range(2):assert np.isclose(aucs[i],roc_auc_score(y,scores[i]),atol=1e-12)
    delta=float(aucs[0]-aucs[1])
    # A patient is drawn once per replicate and all of their samples travel together.
    ids=pd.Index(sorted(set(patients)))
    encoding=ids.get_indexer(patients)
    patient_y=np.array([np.unique(y[encoding==i]).item() for i in range(len(ids))])
    pids=np.flatnonzero(patient_y==1);nids=np.flatnonzero(patient_y==0)
    replicates=[];draws=[]
    for _ in range(2000):
        chosen=np.concatenate([rng.choice(pids,len(pids),replace=True),rng.choice(nids,len(nids),replace=True)])
        draws.append(chosen)
        count=np.bincount(chosen,minlength=len(ids))[encoding]
        pw,nw=count[y==1],count[y==0]
        replicates.append(float(pw@(kernels[0]-kernels[1])@nw/(pw.sum()*nw.sum())))
    bootstrap_draws[dataset]=np.array(draws)
    bootstrap_draws[dataset+'_patients']=np.array(ids,dtype=str)
    bootstrap_draws[dataset+'_delta']=np.array(replicates)
    replicates=np.array(replicates)
    var=float(replicates.var(ddof=1))
    if var<=1e-15:return delta,np.nan,var,len(ids)
    # Center the sampling distribution to approximate the null delta=0.
    p=float((1+np.count_nonzero(np.abs(replicates-delta)>=abs(delta)-1e-12))/(len(replicates)+1))
    return delta,p,var,len(ids)
rows=[]
for ds, split in [('GSE178341','grouped_OOF'),('GSE132465','external'),('GSE200997','external')]:
    a=pred[(pred.dataset==ds)&(pred.split==split)&(pred.model=='18 subtypes + weighted MLH1')].set_index('sampleID').sort_index()
    b=pred[(pred.dataset==ds)&(pred.split==split)&(pred.model=='18 subtypes + raw MLH1')].set_index('sampleID').sort_index()
    assert a.index.equals(b.index) and a.index.is_unique
    assert np.array_equal(a.y,b.y)
    if ds=='GSE200997':
        delta=roc_auc_score(a.y,a.score)-roc_auc_score(b.y,b.score);p=np.nan;var=0.;npatients=15
    else:
        joined=a.reset_index().merge(ledger[ledger.dataset==ds][['sampleID','patientID']],on='sampleID',validate='one_to_one').set_index('sampleID').reindex(a.index)
        assert joined.patientID.notna().all()
        delta,p,var,npatients=patient_bootstrap_p(a.y.to_numpy(),a.score.to_numpy(),b.score.to_numpy(),joined.patientID.to_numpy(),ds)
    rows.append({'dataset':ds,'split':split,'n_samples':len(a),'n_patients':npatients,'positive_samples':int(a.y.sum()),'baseline':'18 subtypes + raw MLH1','delta':delta,'paired_patient_bootstrap_p_two_sided':p,'bootstrap_variance_delta':var,'bootstrap_replicates':2000 if ds!='GSE200997' else 0,'seed':2026,'method':'Two-sided, null-centered outcome-stratified patient-cluster bootstrap; one advancing RNG stream','interpretation':'Post hoc; conditional on archived predictions; unadjusted; feature discovery and CNV uncertainty excluded' if np.isfinite(p) else 'No test: both scores perfectly separate the two-positive cohort; degenerate difference'})
de=pd.DataFrame(rows)
prior=pd.read_csv(ARTICLE/'weighted_scRNA_update/paired_ablation_AUROC_differences.csv')
for r in de.itertuples():
    q=prior[(prior.dataset==r.dataset)&(prior.split==r.split)&(prior.baseline==r.baseline)]
    assert len(q)==1 and np.isclose(q.iloc[0].delta,r.delta,atol=1e-12)
de.to_csv(HERE/'paired_bootstrap_check.csv',index=False)
np.savez_compressed(HERE/'paired_patient_bootstrap_indices.npz',**bootstrap_draws)

counts=pd.read_csv(OLD/'S1_sample_QC_and_counts.csv')
out=[]
for ds,x in counts[counts.included].groupby('dataset'):
    x=x.dropna(subset=['p','mu','W'])
    for u,v in [('W','mu'),('W','p'),('mu','p')]:
        out.append({'dataset':ds,'n':len(x),'analysis':'Spearman correlation','feature':u,'other':v,'value':float(spearmanr(x[u],x[v]).statistic)})
    for ycol in ['W','mu','p']:
        other=[c for c in ['W','mu','p'] if c!=ycol]
        design=np.column_stack([np.ones(len(x)),x[other].to_numpy()])
        target=x[ycol].to_numpy()
        fit=design@np.linalg.lstsq(design,target,rcond=None)[0]
        sse=np.square(target-fit).sum(); sst=np.square(target-target.mean()).sum()
        out.append({'dataset':ds,'n':len(x),'analysis':'VIF with intercept (W, mu, p)','feature':ycol,'other':'+'.join(other),'value':float(sst/sse)})
pd.DataFrame(out).to_csv(HERE/'weighted_feature_dependence.csv',index=False)
manifest={'purpose':'Manuscript-only clarification with post hoc frozen-score tests and label-free feature diagnostics; no refitting, selection or weight changes','new_random_sampling':'Paired patient bootstrap only; seed 2026 initialized once, stream advanced across all resamples and cohorts','sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [PRED,LEDGER,OLD/'S1_sample_QC_and_counts.csv',ARTICLE/'weighted_scRNA_update/paired_ablation_AUROC_differences.csv']},'checks':'Exact sample and label alignment; patients remain together; kernel AUROCs equal sklearn AUROC; differences equal archived paired differences','earlier_nominal_DeLong':'Superseded: sample-level independence is inappropriate for the development cohort with two duplicated patients'}
(HERE/'evidence_checks.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print(de.to_string(index=False))
print(pd.DataFrame(out).to_string(index=False))
