"""Refit or replay all passing subtypes at four source AUROC cutoffs; no count cap."""
from pathlib import Path
import argparse,json,hashlib
import joblib,numpy as np,pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score,balanced_accuracy_score,brier_score_loss
HERE=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--replay',action='store_true');p.add_argument('--output',type=Path)
    args=p.parse_args();out=HERE/'cutoff_analysis'
    if not args.replay:out.mkdir(exist_ok=True);(out/'models').mkdir(exist_ok=True)
    data={ds:pd.read_csv(HERE/'inputs'/f'{ds}.csv',float_precision='round_trip') for ds in ['GSE178341','GSE132465','GSE200997']}
    dev=data['GSE178341'];membership=pd.read_csv(HERE/'inputs/threshold_membership.csv');source=joblib.load(HERE/'models/Joint_foldfull.joblib')
    records=[];models=[]
    for cutoff in [.60,.65,.70,.75]:
        panel=membership[membership.screening_cutoff.eq(cutoff)&membership.in_model].feature.tolist()
        assert len(panel)=={.6:39,.65:24,.7:17,.75:9}[cutoff]
        cols=panel+['weighted_MLH1'];label=f'cutoff{round(cutoff*100)}'
        for fold in [1,2,3,4,5,'full']:
            train=dev if fold=='full' else dev[dev.fold.ne(fold)]
            path=out/'models'/f'{label}_fold{fold}.joblib'
            if args.replay:
                artifact=joblib.load(path);pipe=artifact['pipeline']
                assert artifact['features']==cols and artifact['training_samples']==train.sampleID.tolist()
                np.testing.assert_allclose(pipe.named_steps['scaler'].mean_,train[cols].to_numpy().mean(axis=0),atol=1e-12,rtol=0)
            else:
                pipe=clone(source['pipeline']).fit(train[cols].to_numpy(),train.y.to_numpy())
                artifact={'pipeline':pipe,'features':cols,'seed':2026,'cutoff':cutoff,'n_subtypes':len(panel),'fold':str(fold),'training_samples':train.sampleID.tolist(),'training_patients':train.patientID.tolist(),'selection':'All passing source oriented AUROC; no count cap; fixed source direction'}
                joblib.dump(artifact,path)
            models.append({'cutoff':cutoff,'fold':str(fold),'path':str(path.relative_to(HERE)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            targets=[('GSE178341',dev[dev.fold.eq(fold)],'grouped_OOF')] if fold!='full' else [(ds,d,'training_apparent' if ds=='GSE178341' else 'external') for ds,d in data.items()]
            for ds,frame,split in targets:
                if split=='grouped_OOF':assert not set(train.patientID)&set(frame.patientID)
                scores=pipe.predict_proba(frame[cols].to_numpy())[:,1]
                records.extend(dict(cutoff=cutoff,n_subtypes=len(panel),dataset=ds,split=split,fold=str(fold),sampleID=s,y=int(y),score=float(v)) for s,y,v in zip(frame.sampleID,frame.y,scores))
    pred=pd.DataFrame(records);rows=[]
    expected=pd.read_csv(HERE/'tables/threshold_performance.csv')
    for (cutoff,ds,split),part in pred.groupby(['cutoff','dataset','split']):
        row=dict(cutoff=cutoff,dataset=ds,split=split,AUROC=roc_auc_score(part.y,part.score),BA=balanced_accuracy_score(part.y,part.score>=.5),Brier=brier_score_loss(part.y,part.score))
        if split!='training_apparent':
            ref=expected[expected.screening_cutoff.eq(cutoff)&expected.dataset.eq(ds)].iloc[0]
            np.testing.assert_allclose([row[c] for c in ['AUROC','BA','Brier']],[ref[c] for c in ['AUROC','BA','Brier']],atol=1e-12,rtol=0)
        rows.append(row)
    if args.replay:
        saved=pd.read_csv(out/'predictions.csv',float_precision='round_trip');np.testing.assert_allclose(saved.score,pred.score,atol=1e-12,rtol=0)
    else:
        pred.to_csv(out/'predictions.csv',index=False);pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False);pd.DataFrame(models).to_csv(out/'model_index.csv',index=False)
    development=pd.DataFrame(rows).query("dataset=='GSE178341' and split=='grouped_OOF'")
    assert development.loc[development.AUROC.idxmax(),'cutoff']==.7
    cnv=pd.read_csv(HERE/'tables/CNV_sensitivity_predictions.csv');cm=pd.read_csv(HERE/'tables/CNV_sensitivity_metrics.csv')
    for row in cm.itertuples():
        z=cnv[cnv['quantile'].eq(row.quantile)&cnv['mode'].eq(row.mode)&cnv.dataset.eq(row.dataset)&cnv.split.eq(row.split)]
        assert abs(roc_auc_score(z.y,z.score)-row.AUROC)<1e-12
    proof={'passed':True,'mode':'Saved-model replay' if args.replay else 'Actual fresh fit','cutoff_models':len(models),'predictions':len(pred),'counts':[39,24,17,9],'selected_cutoff':.70,'selection_uses':'Development grouped OOF only; final choice after comparison','all12_reported_cohort_AUROC_BA_Brier_reproduced':True,'CNV_metric_checks':True,'seed':2026}
    target=args.output or out/'verification.json';target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(proof,indent=2));print(json.dumps(proof,indent=2))
if __name__=='__main__':main()
