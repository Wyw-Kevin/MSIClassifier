"""Replay actual panel12 checkpoints and archived cohort-held-out MMR4 locks.

No feature selection, threshold search, refit or fusion-weight optimization.
Inputs are archived measurements, not fresh deconvolution of raw expression.
"""
from pathlib import Path
import argparse, hashlib, json
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
GENES = ['MLH1','MSH2','MSH6','PMS2']
DEV = ['GSE39582','TCGA-COADREAD']

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def replay(snapshot):
    snapshot = Path(snapshot)
    raw = pd.read_csv(snapshot/'inputs/all_cohort_features.csv',float_precision='round_trip')
    if raw.duplicated(['dataset','sampleID']).any(): raise ValueError('Duplicate sample key')
    if len(raw)!=1212: raise ValueError('Archived sample set must contain1212 specimens')
    results=[]; sources={}; scaler_errors=[]
    for cohort, part in raw.groupby('dataset',sort=True):
        part=part.copy()
        lock_path=snapshot/'source_archive'/f'heldout_{cohort}.json'
        lock=json.loads(lock_path.read_text())
        if cohort in lock['training_cohorts'] or lock['held_out_labels_used_for_fit']:
            raise ValueError('Held-out cohort entered its own threshold fit')
        expression=part[[g+'_raw' for g in GENES]].to_numpy(float)
        mu=np.asarray(lock['held_out_scaler']['mean'])
        sd=np.asarray(lock['held_out_scaler']['sample_sd'])
        np.testing.assert_allclose(expression.mean(axis=0),mu,rtol=0,atol=1e-12)
        np.testing.assert_allclose(expression.std(axis=0,ddof=1),sd,rtol=0,atol=1e-12)
        scaler_errors.append(float(np.max(np.abs(expression.std(axis=0,ddof=1)-sd))))
        part['MMR4_LOCO']=expit((np.asarray(lock['cutoffs'])-(expression-mu)/sd)/.5).max(axis=1)
        part['Subtype']=np.nan
        for fold in sorted(part.fold.unique()):
            model_path=snapshot/'models'/f'panel12_fold{int(fold)}.joblib'
            artifact=joblib.load(model_path)
            if artifact['seed']!=2026 or artifact['panel']!='panel12': raise ValueError('Wrong model identity')
            mask=part.fold.eq(fold)
            x=part.loc[mask,artifact['features']].to_numpy(float)
            if not np.isfinite(x).all() or (x<0).any(): raise ValueError('Invalid subtype input')
            part.loc[mask,'Subtype']=artifact['pipeline'].predict_proba(x)[:,1]
            sources[str(model_path.relative_to(snapshot))]=sha(model_path)
        part['Fusion05']=.5*part.Subtype+.5*part.MMR4_LOCO
        part['split']='development_OOF' if cohort in DEV else 'retrospective'
        results.append(part[['dataset','sampleID','y','scope','fold','split','Subtype','MMR4_LOCO','Fusion05']])
        sources[str(lock_path.relative_to(snapshot))]=sha(lock_path)
    result=pd.concat(results,ignore_index=True).sort_values(['dataset','sampleID']).reset_index(drop=True)
    saved=pd.read_csv(snapshot/'predictions.csv',float_precision='round_trip')
    saved=saved[(saved.panel=='panel12') & (saved.split!='development_apparent')].sort_values(['dataset','sampleID']).reset_index(drop=True)
    np.testing.assert_array_equal(result[['dataset','sampleID','y']],saved[['dataset','sampleID','y']])
    columns=['Subtype','MMR4_LOCO','Fusion05']
    error=float(np.max(np.abs(result[columns].to_numpy()-saved[columns].to_numpy())))
    np.testing.assert_allclose(result[columns],saved[columns],rtol=0,atol=1e-12)
    metrics=[{'dataset':c,'split':p.split.iloc[0],'component':s,'n':len(p),'MSI':int(p.y.sum()),'MSS':int((1-p.y).sum()),'AUROC':float(roc_auc_score(p.y,p[s]))} for c,p in result.groupby('dataset',sort=True) for s in columns]
    m=pd.DataFrame(metrics)
    macro=[{'scope':scope,'component':s,'n_cohorts':len(cohorts),'AUROC':float(m[m.dataset.isin(cohorts)&m.component.eq(s)].AUROC.mean())} for scope,cohorts in [('development_OOF',DEV),('retrospective',sorted(set(raw.dataset)-set(DEV))),('all_eight',sorted(set(raw.dataset)))] for s in columns]
    report={'status':'passed','n_samples':len(result),'n_cohorts':8,'model_refitted':False,'MMR4_refitted':False,'max_score_error':error,'max_scaler_sd_error':max(scaler_errors),'score_unit':'bulk specimen','fusion_weights':[.5,.5],'macro_results':macro,'cohort_results':metrics,'source_sha256':sources,'input_sha256':sha(snapshot/'inputs/all_cohort_features.csv'),'interpretation':'Replay of fixed archived measurements/models and eight LOCO locks; no independent validation'}
    return result,report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=HERE/'outputs/frozen_bulk')
    args=parser.parse_args()
    result,report=replay(HERE/'classifier_snapshot')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.output_dir/'predictions.csv',index=False,float_format='%.17g')
    pd.DataFrame(report['cohort_results']).to_csv(args.output_dir/'cohort_metrics.csv',index=False,float_format='%.17g')
    pd.DataFrame(report['macro_results']).to_csv(args.output_dir/'macro_metrics.csv',index=False,float_format='%.17g')
    (args.output_dir/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('PASS:',len(result),'bulk samples; maximum archived-score error',report['max_score_error'])

if __name__=='__main__': main()
