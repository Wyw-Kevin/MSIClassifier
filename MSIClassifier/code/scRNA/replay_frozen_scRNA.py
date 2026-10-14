"""Relocatable archived-feature/model, S7, paired bootstrap and VIF replay; no refitting."""
from pathlib import Path
import argparse,json,hashlib,platform
import numpy as np,pandas as pd,joblib,sklearn
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from package_paths import project_root

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path)
    p.add_argument('--output-dir',type=Path)
    args=p.parse_args();root=project_root(args.project_root);f=root/'最终模型';art=f/'report/article'
    b=f/'data/scRNA/result/diagnostics/shared18_malignant_fraction_weighted_mlh1_seed2026'
    sup=art/'bulk12_loco_manuscript_seed2026_20261013/supplement'
    pred=pd.read_csv(art/'weighted_scRNA_update/all_model_predictions.csv',float_precision='round_trip')
    model=joblib.load(f/'model/reference/shared18_weighted_mlh1_2026/model.joblib')
    assert model['seed']==2026 and model['formula']=='mean_MLH1 * N_malignant / N_epi' and len(model['features'])==19
    features=model['features']; rows=[];score_error=0.
    for ds in ['GSE178341','GSE132465','GSE200997']:
        t=pd.read_csv(b/f'{ds}_malignant_fraction_weighted_MLH1_refit_features.csv',float_precision='round_trip').set_index('sampleID')
        split='training_apparent' if ds=='GSE178341' else 'external'
        q=pred[pred.dataset.eq(ds)&pred.split.eq(split)&pred.model.eq('18 subtypes + weighted MLH1')].set_index('sampleID')
        assert set(t.index)==set(q.index) and np.isfinite(t[features].to_numpy()).all()
        s=model['pipeline'].predict_proba(t.loc[q.index,features])[:,1]
        err=float(np.max(np.abs(s-q.score.to_numpy())));assert err<1e-12;score_error=max(score_error,err)
        rows.append({'dataset':ds,'split':split,'n':len(q),'AUROC':float(roc_auc_score(q.y,s)),'max_frozen_score_error':err})
    ledger=pd.read_csv(f/'data/scRNA/result/diagnostics/shared18_same_sample_benchmark_seed2026/sample_ledger.csv')
    d=ledger[ledger.dataset.eq('GSE178341')];assert len(d)==64 and d.patientID.nunique()==62 and d.groupby('patientID').fold.nunique().eq(1).all()
    controls=pd.read_csv(art/'revision_review_20260928/additional_control_predictions.csv',float_precision='round_trip')
    allpred=pd.concat([pred,controls],ignore_index=True);s7=pd.read_csv(sup/'S7.csv',float_precision='round_trip')
    for r in s7.itertuples():
        q=allpred[allpred.dataset.eq(r.dataset)&allpred.split.eq(r.split)&allpred.model.eq(r.model)]
        assert q.sampleID.is_unique and len(q)==r.n and np.isclose(roc_auc_score(q.y,q.score),r.AUROC,atol=1e-12)
    draws=np.load(sup/'paired_patient_bootstrap_indices.npz',allow_pickle=False);rng=np.random.default_rng(2026);boot=[]
    for ds,split in [('GSE178341','grouped_OOF'),('GSE132465','external')]:
        q=pred[pred.dataset.eq(ds)&pred.split.eq(split)]
        wide=q.pivot(index='sampleID',columns='model',values='score').sort_index();y=q.drop_duplicates('sampleID').set_index('sampleID').y.reindex(wide.index).to_numpy()
        ids=pd.Index(draws[ds+'_patients']);patient=ledger[ledger.dataset.eq(ds)].set_index('sampleID').patientID.reindex(wide.index).to_numpy();enc=ids.get_indexer(patient)
        py=np.array([np.unique(y[enc==i]).item() for i in range(len(ids))]);posids=np.flatnonzero(py==1);negids=np.flatnonzero(py==0)
        fresh=np.array([np.concatenate([rng.choice(posids,len(posids),replace=True),rng.choice(negids,len(negids),replace=True)]) for _ in range(2000)]);assert np.array_equal(fresh,draws[ds])
        a=np.vstack([wide['18 subtypes + weighted MLH1'],wide['18 subtypes + raw MLH1']]);pos=a[:,y==1];neg=a[:,y==0];k=(pos[:,:,None]>neg[:,None,:]).astype(float)+.5*(pos[:,:,None]==neg[:,None,:])
        delta=float(roc_auc_score(y,a[0])-roc_auc_score(y,a[1]));reps=[]
        for chosen in draws[ds]:
            c=np.bincount(chosen,minlength=len(ids))[enc];pw,nw=c[y==1],c[y==0];reps.append(float(pw@(k[0]-k[1])@nw/(pw.sum()*nw.sum())))
        reps=np.array(reps);assert np.allclose(reps,draws[ds+'_delta'],rtol=0,atol=1e-14)
        pv=float((1+np.count_nonzero(np.abs(reps-delta)>=abs(delta)-1e-12))/(len(reps)+1))
        saved=pd.read_csv(sup/'S6_Bootstrap_p.csv');assert np.isclose(pv,saved[saved.dataset.eq(ds)].iloc[0].paired_patient_bootstrap_p_two_sided,atol=1e-12)
        boot.append({'dataset':ds,'patient_clusters':len(ids),'replicates':2000,'delta':delta,'p_two_sided_null_centered':pv,'saved_index_replay':True})
    counts=pd.read_csv(art/'revision_review_20260928/S1_sample_QC_and_counts.csv');expected=pd.read_csv(sup/'S7_Correlation_VIF.csv');dep=[]
    for ds,t in counts[counts.included].groupby('dataset'):
        t=t.dropna(subset=['p','mu','W'])
        for u,v in [('W','mu'),('W','p'),('mu','p')]:dep.append(dict(dataset=ds,n=len(t),analysis='Spearman correlation',feature=u,other=v,value=float(spearmanr(t[u],t[v]).statistic)))
        for feature in ['W','mu','p']:
            others=[c for c in ['W','mu','p'] if c!=feature];X=np.column_stack([np.ones(len(t)),t[others].to_numpy()]);y=t[feature].to_numpy();fit=X@np.linalg.lstsq(X,y,rcond=None)[0]
            dep.append(dict(dataset=ds,n=len(t),analysis='VIF with intercept (W, mu, p)',feature=feature,other='+'.join(others),value=float(np.square(y-y.mean()).sum()/np.square(y-fit).sum())))
    pd.testing.assert_frame_equal(expected,pd.DataFrame(dep),check_exact=False,rtol=1e-10,atol=1e-10)
    # Original primary fold models were not saved. Audit separately identified q95 sensitivity refits.
    sense=art/'scRNA_AUROC_threshold_all_candidates_seed2026_20261013';dev=pd.read_csv(sense/'inputs/GSE178341_q95_features_labels_folds.csv',float_precision='round_trip').set_index('sampleID')
    primary_oof=pred[pred.dataset.eq('GSE178341')&pred.split.eq('grouped_OOF')&pred.model.eq('18 subtypes + weighted MLH1')].set_index('sampleID');oof_error=0.
    for fold in sorted(dev.fold.unique()):
        a=joblib.load(sense/f'models/CNV_q95_refit__fold{int(fold)}.joblib');q=dev[dev.fold.eq(fold)]
        assert not set(a['training_patients'])&set(q.patientID)
        replay=a['pipeline'].predict_proba(q[a['features']].to_numpy())[:,1]
        oof_error=max(oof_error,float(np.max(np.abs(replay-primary_oof.loc[q.index,'score'].to_numpy()))))
    assert oof_error<1e-12
    result={'status':'verified','refitting':False,'project_root':str(root),'frozen_full_model_replay':rows,'max_frozen_score_error':score_error,'archived_primary_OOF_AUROC':float(roc_auc_score(primary_oof.y,primary_oof.score)),'OOF_replay_limit':'Original weighted five fold artifacts unavailable; separate archived CNV_q95 sensitivity refits numerically reproduce the primary saved OOF scores without relabelling their fit identity.','max_separately_identified_q95_fold_refit_error':oof_error,'S7_archived_score_rows_verified':len(s7),'paired_patient_bootstrap':boot,'GSE200997_p_test':'Skipped: degenerate paired delta; zero p-test draws','correlation_VIF_rows_verified':len(dep),'software':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__}}
    if args.output_dir:
        args.output_dir.mkdir(parents=True,exist_ok=True);(args.output_dir/'replay_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
