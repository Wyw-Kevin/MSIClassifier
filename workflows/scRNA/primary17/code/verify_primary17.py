"""Replay current17 checkpoints and verify patient-cluster uncertainty from saved inputs."""
from pathlib import Path
import argparse,json,hashlib
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
HERE=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=HERE/'verification');p.add_argument('--output',type=Path)
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    pred=pd.read_csv(HERE/'tables/all_predictions.csv',float_precision='round_trip')
    metric=pd.read_csv(HERE/'tables/metrics.csv',float_precision='round_trip')
    deltas=pd.read_csv(HERE/'tables/paired_differences.csv',float_precision='round_trip')
    draws=np.load(HERE/'tables/patient_bootstrap_indices.npz',allow_pickle=False)
    maximum=0.;maximum_interval=0.;n_prediction=0
    for ds in ['GSE178341','GSE132465','GSE200997']:
        frame=pd.read_csv(HERE/'tables'/f'{ds}_primary17_features.csv',float_precision='round_trip').sort_values('sampleID')
        assert frame.groupby('patientID').y.nunique().eq(1).all()
        if ds=='GSE178341':assert len(frame)==64 and frame.patientID.nunique()==62 and frame.groupby('patientID').fold.nunique().eq(1).all()
        ids=pd.Index(draws[ds+'_patients']);codes=ids.get_indexer(frame.patientID)
        counts=np.array([np.bincount(row,minlength=len(ids))[codes] for row in draws[ds]])
        for split in (['grouped_OOF','training_apparent'] if ds=='GSE178341' else ['external']):
            part=pred[pred.dataset.eq(ds)&pred.split.eq(split)]
            wide=part.pivot(index='sampleID',columns='model',values='score').reindex(frame.sampleID)
            boot={}
            for name in wide.columns:
                actual=np.full(len(frame),np.nan)
                for fold in (range(1,6) if split=='grouped_OOF' else ['full']):
                    m=joblib.load(HERE/'models'/f'{name}_fold{fold}.joblib')
                    mask=frame.fold.eq(fold).to_numpy() if split=='grouped_OOF' else np.ones(len(frame),bool)
                    if split=='grouped_OOF':assert not set(m['training_patients'])&set(frame.loc[mask,'patientID'])
                    actual[mask]=m['pipeline'].predict_proba(frame.loc[mask,m['features']].to_numpy())[:,1]
                err=float(np.max(np.abs(actual-wide[name].to_numpy())));maximum=max(maximum,err);assert err<1e-12
                n_prediction+=len(actual)
                y=frame.y.to_numpy();values=wide[name].to_numpy()
                # Independent verification uses sklearn's weighted ROC routine, not the computation's rank kernel.
                rep=np.array([roc_auc_score(y,values,sample_weight=w) for w in counts])
                boot[name]=rep;ref=metric[metric.dataset.eq(ds)&metric.split.eq(split)&metric.model.eq(name)].iloc[0]
                assert abs(roc_auc_score(y,values)-ref.AUROC)<1e-12
                err=float(np.max(np.abs(np.quantile(rep,[.025,.975])-[ref.CI_low,ref.CI_high])))
                maximum_interval=max(maximum_interval,err);assert err<1e-12
            for row in deltas[deltas.dataset.eq(ds)&deltas.split.eq(split)].itertuples():
                reps=boot['Joint']-boot[row.baseline]
                np.testing.assert_allclose(np.quantile(reps,[.025,.975]),[row.CI_low,row.CI_high],atol=1e-12,rtol=0)
                if np.ptp(reps)>1e-12:
                    pv=(1+np.count_nonzero(np.abs(reps-row.delta)>=abs(row.delta)-1e-12))/2001
                    assert abs(pv-row.p_two_sided_null_centered)<1e-12
    cnv=pd.read_csv(HERE/'tables/CNV_sensitivity_predictions.csv',float_precision='round_trip')
    cm=pd.read_csv(HERE/'tables/CNV_sensitivity_metrics.csv',float_precision='round_trip')
    for row in cm.itertuples():
        z=cnv[cnv['quantile'].eq(row.quantile)&cnv['mode'].eq(row.mode)&cnv.dataset.eq(row.dataset)&cnv.split.eq(row.split)]
        assert len(z)==row.n and abs(roc_auc_score(z.y,z.score)-row.AUROC)<1e-12
    panel=pd.read_csv(HERE/'tables/selected17_features.csv').feature.tolist()
    membership=pd.read_csv(HERE/'inputs/threshold_membership.csv')
    selected=membership[membership.screening_cutoff.eq(.7)&membership.in_model]
    assert panel==selected.feature.tolist() and selected.reference_oriented_AUROC.gt(.7).all() and len(panel)==17 and 'cM04' not in panel
    screen=pd.read_csv(HERE/'tables/threshold_performance.csv');dev=screen[screen.dataset.eq('GSE178341')]
    assert len(dev)==4 and dev.loc[dev.AUROC.idxmax(),'screening_cutoff']==.7
    proof={'passed':True,'primary_subtypes':17,'scored_predictions':n_prediction,'max_prediction_error':maximum,'max_interval_error':maximum_interval,
           'bootstrap_check':'All2000 paired patient-cluster draws independently evaluated with sklearn sample_weight ROC',
           'selection':'Highest development grouped OOF of four cutoffs; same data used for selection; no cutoff-selection uncertainty in CI',
           'CNV_score_metrics_verified':True,'patient_fold_separation_verified':True,'seed':2026}
    output=args.output or args.output_dir/'verification.json';output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(proof,indent=2))
    print(json.dumps(proof,indent=2))
if __name__=='__main__':main()
