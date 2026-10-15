"""Frozen q95-classifier diagnostic: include vs exclude exact CNV-boundary ties.

Uses audited primary candidate-cell MLH1 values and the unchanged epithelial
denominator. This is input perturbation, with no model refit or cutoff selection.
"""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd,joblib
from sklearn.metrics import roc_auc_score,balanced_accuracy_score,brier_score_loss,confusion_matrix
HERE=Path(__file__).resolve().parent
SNAP=HERE.parent/'primary17'

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=HERE/'outputs')
    args=p.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    definitions=pd.read_csv(HERE/'inputs/boundary_definitions.csv',float_precision='round_trip')
    primary=pd.read_csv(SNAP/'tables/all_predictions.csv',float_precision='round_trip')
    allrows=[];summary=[];metrics=[];max_baseline_error=0.
    for d in definitions.itertuples():
        ds=d.dataset
        cells=pd.read_csv(HERE/'inputs'/(ds+'_q95_candidate_cells.csv.gz'),float_precision='round_trip')
        frame=pd.read_csv(HERE/'inputs'/(ds+'_features.csv'),float_precision='round_trip')
        assert len(cells)==d.n_primary_candidates and cells.cnv_score.ge(d.q95_threshold-1e-12).all()
        strict=cells[cells.cnv_score.gt(d.q95_threshold)]
        assert len(strict)==d.n_strict_gt_candidates
        g=cells.groupby('sampleID').MLH1.agg(['size','sum']).reindex(frame.sampleID)
        h=strict.groupby('sampleID').MLH1.agg(['size','sum']).reindex(frame.sampleID)
        primary_w=g['sum'].to_numpy()/frame.n_epithelial.to_numpy()
        np.testing.assert_allclose(primary_w,frame.weighted_MLH1,atol=2e-12,rtol=0)
        assert np.array_equal(g['size'].to_numpy(),frame.n_candidate)
        # Missing candidate sets remain missing; there is no zero-expression imputation.
        n_strict=h['size'].fillna(0).to_numpy().astype(int)
        strict_sum=h['sum'].to_numpy();strict_w=strict_sum/frame.n_epithelial.to_numpy()
        strict_w[n_strict==0]=np.nan
        mu_strict=strict_sum/n_strict
        np.testing.assert_allclose(strict_w,mu_strict*n_strict/frame.n_epithelial.to_numpy(),atol=1e-12,rtol=0,equal_nan=True)
        records=frame[['dataset','sampleID','patientID','y']].copy()
        records['split']='grouped_OOF' if ds=='GSE178341' else 'external'
        records['threshold']=d.q95_threshold;records['n_epithelial']=frame.n_epithelial
        records['n_primary']=frame.n_candidate;records['n_strict_gt']=n_strict
        records['n_tied_removed']=frame.n_candidate-n_strict
        records['primary_W']=frame.weighted_MLH1;records['strict_gt_W']=strict_w
        records['delta_W']=strict_w-frame.weighted_MLH1
        records['primary_score']=np.nan;records['strict_gt_score']=np.nan
        records['scoreable_strict_gt']=np.isfinite(strict_w)
        for fold in (range(1,6) if ds=='GSE178341' else ['full']):
            artifact=joblib.load(SNAP/'models'/f'Joint_fold{fold}.joblib')
            mask=frame.fold.eq(fold).to_numpy() if ds=='GSE178341' else np.ones(len(frame),bool)
            if ds=='GSE178341':assert not set(artifact['training_patients'])&set(frame.loc[mask,'patientID'])
            cols=artifact['features'];pipe=artifact['pipeline']
            records.loc[mask,'primary_score']=pipe.predict_proba(frame.loc[mask,cols].to_numpy())[:,1]
            eligible=mask&np.isfinite(strict_w)
            X=frame.loc[eligible,cols].copy();X['weighted_MLH1']=strict_w[eligible]
            if len(X):records.loc[eligible,'strict_gt_score']=pipe.predict_proba(X.to_numpy())[:,1]
        expected=primary[primary.dataset.eq(ds)&primary.model.eq('Joint')&primary.split.eq(records['split'].iloc[0])].set_index('sampleID').loc[records.sampleID]
        error=float(np.max(np.abs(records.primary_score.to_numpy()-expected.score.to_numpy())));max_baseline_error=max(max_baseline_error,error);assert error<1e-12
        records['delta_score']=records.strict_gt_score-records.primary_score
        records['primary_prediction05']=records.primary_score.ge(.5).astype(int)
        records['strict_gt_prediction05']=np.where(records.scoreable_strict_gt,records.strict_gt_score.ge(.5).astype(int),np.nan)
        records['changed_prediction05']=np.where(records.scoreable_strict_gt,records.primary_prediction05.ne(records.strict_gt_prediction05),np.nan)
        common=records[records.scoreable_strict_gt]
        for label,column in [('Primary_ge_common','primary_score'),('Strict_gt_common','strict_gt_score')]:
            y=common.y.to_numpy();s=common[column].to_numpy();tn,fp,fn,tp=confusion_matrix(y,s>=.5,labels=[0,1]).ravel()
            metrics.append({'dataset':ds,'comparison':label,'n':len(common),'positive':int(y.sum()),'AUROC':roc_auc_score(y,s) if len(np.unique(y))==2 else np.nan,
                'BA':balanced_accuracy_score(y,s>=.5),'Brier':brier_score_loss(y,s),'TN':tn,'FP':fp,'FN':fn,'TP':tp,
                'scope':'Same scoreable samples; frozen q95 classifier; no refit; no external-based threshold selection'})
        summary.append({'dataset':ds,'threshold':d.q95_threshold,'n_primary_candidates':len(cells),'n_strict_gt_candidates':len(strict),'n_boundary_cells':len(cells)-len(strict),
            'boundary_fraction':(len(cells)-len(strict))/len(cells),'primary_scoreable_samples':len(frame),'strict_gt_scoreable_samples':len(common),
            'newly_unscoreable_samples':';'.join(records.loc[~records.scoreable_strict_gt,'sampleID']),
            'samples_with_W_change':int(common.delta_W.abs().gt(1e-12).sum()),'max_abs_W_change':float(common.delta_W.abs().max()),
            'max_abs_score_change':float(common.delta_score.abs().max()),'classification_changes_at05':int(common.changed_prediction05.sum())})
        allrows.append(records)
    values=pd.concat(allrows,ignore_index=True);metrics=pd.DataFrame(metrics);summary=pd.DataFrame(summary)
    values.to_csv(out/'S3_boundary_sample_features_scores.csv',index=False);summary.to_csv(out/'S2_boundary_summary.csv',index=False);metrics.to_csv(out/'S3_boundary_frozen_AUROC.csv',index=False)
    proof={'passed':True,'seed':2026,'random_resampling_used':False,'q95_model_frozen':True,'classifier_refit':False,'primary_predictions_max_error':max_baseline_error,
        'strict_rule':'cnv_score > same unrounded q95 threshold; equality excluded','primary_rule':'cnv_score >= q95 -1e-12; tolerance adds no cells in saved audit',
        'missing_handling':'No candidate cells => W/score missing; paired metrics use common scoreable specimens','external_outcomes_used_for_selection':False,'interpretation':'Boundary-rule input perturbation only; primary q95 rule unchanged'}
    (out/'verification.json').write_text(json.dumps(proof,indent=2));print(summary.to_string(index=False));print(metrics[['dataset','comparison','n','AUROC','BA','FP','FN']].to_string(index=False))
if __name__=='__main__':main()
