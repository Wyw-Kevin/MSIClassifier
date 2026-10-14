"""Label-free full-development panel12 inference; optional matching LOCO lock."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd,joblib
from scipy.special import expit

HERE=Path(__file__).resolve().parent
GENES=['MLH1','MSH2','MSH6','PMS2']

def score(table,model,lock=None):
    if model.get('panel')!='panel12' or model.get('fold')!=0 or model.get('seed')!=2026:
        raise ValueError('Use the actual full-development panel12 model')
    features=model['features'];missing=[f for f in features if f not in table]
    if len(features)!=12 or missing:raise ValueError('Missing panel12 features: '+', '.join(missing))
    keys=['dataset','sampleID'] if 'dataset' in table else ['sampleID']
    if 'sampleID' not in table or table.empty or table[keys].isna().any().any() or table.duplicated(keys).any():
        raise ValueError('Require nonempty rows with unique sample keys')
    x=table[features].to_numpy(dtype=float)
    if not np.isfinite(x).all() or (x<0).any() or (x>1+1e-12).any():
        raise ValueError('Subtype inputs must be finite parent-normalized fractions in[0,1]')
    result=table[keys].copy();result['Subtype']=model['pipeline'].predict_proba(x)[:,1]
    if lock is not None:
        cohort=lock.get('held_out_cohort')
        if 'dataset' not in table or set(table.dataset)!={cohort}:
            raise ValueError('The supplied LOCO lock belongs to a different cohort; no universal MMR4 deployment lock is supplied')
        if lock.get('held_out_labels_used_for_fit') or cohort in lock.get('training_cohorts',[]):
            raise ValueError('Invalid held-out-cohort threshold provenance')
        cols=[g+'_raw' for g in GENES]
        if any(c not in table for c in cols):raise ValueError('MMR4 requires all four pre-Z-score log2 expression columns')
        raw=table[cols].to_numpy(dtype=float)
        mu=np.asarray(lock['held_out_scaler']['mean'],dtype=float)
        sd=np.asarray(lock['held_out_scaler']['sample_sd'],dtype=float)
        cutoff=np.asarray(lock['cutoffs'],dtype=float)
        if any(a.shape!=(4,) for a in [mu,sd,cutoff]) or not all(np.isfinite(a).all() for a in [raw,mu,sd,cutoff]) or (sd<=0).any():
            raise ValueError('Invalid log inputs or frozen cohort scaler/cutoff lock')
        result['MMR4_LOCO']=expit((cutoff-(raw-mu)/sd)/.5).max(axis=1)
        result['Fusion05']=.5*result.Subtype+.5*result.MMR4_LOCO
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--feature-table',type=Path,required=True)
    p.add_argument('--model',type=Path,default=HERE/'classifier_snapshot/models/panel12_fold0.joblib')
    p.add_argument('--mmr4-lock',type=Path,help='Explicit matching existing cohort lock; no default across cohorts')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();data=pd.read_csv(args.feature_table,float_precision='round_trip')
    lock=json.loads(args.mmr4_lock.read_text()) if args.mmr4_lock else None
    result=score(data,joblib.load(args.model),lock)
    args.output.parent.mkdir(parents=True,exist_ok=True);result.to_csv(args.output,index=False,float_format='%.17g')
    print(json.dumps({'n':len(result),'model_identity':'full-development panel12; no labels/refit','fusion':lock is not None,'output':str(args.output)}))

if __name__=='__main__':main()
