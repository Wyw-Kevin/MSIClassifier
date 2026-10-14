"""Independent checkpoint, provenance, score and metric verification."""
from pathlib import Path
import json, hashlib, sys
import numpy as np
import pandas as pd
import joblib
from scipy.special import expit

HERE=Path(__file__).resolve().parent
OUT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else HERE
checks=[]

def check(name,condition,detail=None):
    if not bool(condition): raise AssertionError((name,detail))
    checks.append({'check':name,'passed':True,'detail':detail})

def close(name,a,b,tol=1e-12):
    error=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
    check(name,error<tol,{'max_absolute_difference':error,'tolerance':tol})

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def auc(y,p):
    positive=np.asarray(p)[np.asarray(y)==1];negative=np.asarray(p)[np.asarray(y)==0]
    comparisons=positive[:,None]-negative[None,:]
    return float(((comparisons>0)+.5*(comparisons==0)).mean())

def metric_values(y,p):
    y=np.asarray(y);p=np.asarray(p);pred=p>=.5
    sensitivity=pred[y==1].mean();specificity=(~pred[y==0]).mean()
    return [auc(y,p),(sensitivity+specificity)/2,sensitivity,specificity,float(((p-y)**2).mean())]

def main():
    raw=pd.read_csv(OUT/'inputs/all_cohort_features.csv')
    pred=pd.read_csv(OUT/'predictions.csv')
    metrics=pd.read_csv(OUT/'cohort_metrics.csv')
    macro=pd.read_csv(OUT/'macro_metrics.csv')
    panels=json.loads((OUT/'panels_locked_before_fit.json').read_text())
    manifest=json.loads((OUT/'source_manifest.json').read_text())
    check('source count',len(manifest)==19)
    for item in manifest:check('archived source hash '+item['archive'],sha(OUT/'source_archive'/item['archive'])==item['sha256'])
    protocol=json.loads((OUT/'protocol.json').read_text())
    check('explicit seed and estimator',protocol['seed']==2026 and protocol['classifier']['random_state']==2026)
    check('locked fixed fusion',protocol['fusion_weights']==[.5,.5])
    check('prediction identities',len(pred)==4208 and not pred.duplicated(['panel','split','dataset','sampleID']).any())
    check('finite bounded scores',np.isfinite(pred[['Subtype','MMR4_LOCO','Fusion05']]).all().all() and pred[['Subtype','MMR4_LOCO','Fusion05']].ge(0).all().all() and pred[['Subtype','MMR4_LOCO','Fusion05']].le(1).all().all())
    close('equal fusion identity',pred.Fusion05,.5*pred.Subtype+.5*pred.MMR4_LOCO)
    source=OUT/'source_archive'
    ledger=pd.read_csv(source/'ledger.csv').set_index('feature')
    candidates=ledger.index[ledger.auc_best_direction>.70].tolist()
    check('17 source candidates',len(candidates)==17)
    table=pd.read_csv(OUT/'panel_membership.csv').set_index('feature')
    independently_selected={}
    for panel,threshold in [('panel7',.6),('panel12',.5)]:
        selected=[]
        for f in candidates:
            values=[]
            for cohort in ['GSE39582','TCGA-COADREAD']:
                part=raw[raw.dataset==cohort]
                values.append(auc(part.y,part[f]*ledger.loc[f,'direction_sign']))
                close('oriented screening '+f+' '+cohort,values[-1],table.loc[f,cohort+'_oriented_AUROC'])
            if all(v>threshold for v in values):selected.append(f)
        independently_selected[panel]=selected
        check('membership '+panel,set(selected)==set(panels[panel]) and len(selected)==int(panel.replace('panel','')))
    check('nested feature panels',set(panels['panel7'])<set(panels['panel12']))
    expected_extra={'cS18','cTNI04','cS17','cM05','cTNI08'}
    check('five additions',set(panels['panel12'])-set(panels['panel7'])==expected_extra)
    dev=raw[raw.scope=='development']
    all_pred=pd.concat([pred,pd.read_csv(OUT/'historical10_runtime_predictions.csv')],ignore_index=True)
    fit_audit=json.loads((OUT/'fit_audit.json').read_text())
    check('18 fitted checkpoints',len(fit_audit)==18)
    for record in fit_audit:
        file=OUT/record['checkpoint'];model=joblib.load(file);fold=record['fold'];panel=record['panel'];features=model['features']
        check('checkpoint hash '+file.name,sha(file)==record['checkpoint_sha256'])
        train=dev if fold==0 else dev[dev.fold!=fold]
        check('training identities '+file.name,train[['dataset','sampleID']].to_dict('records')==model['training_keys'])
        check('training counts '+file.name,len(train)==record['n_train'] and int(train.y.sum())==record['n_positive'])
        if fold:
            holdout=dev[dev.fold==fold]
            check('holdout exclusion '+file.name,not set(map(tuple,train[['dataset','sampleID']].to_numpy()))&set(map(tuple,holdout[['dataset','sampleID']].to_numpy())))
        pipeline=model['pipeline'];scaler=pipeline.named_steps['scaler'];mlp=pipeline.named_steps['mlp']
        close('train scaler means '+file.name,scaler.mean_,train[features].to_numpy().mean(axis=0))
        close('train scaler variance '+file.name,scaler.var_,train[features].to_numpy().var(axis=0,ddof=0))
        check('model configuration '+file.name,mlp.random_state==2026 and mlp.hidden_layer_sizes==(4,) and mlp.alpha==1 and not mlp.early_stopping)
        rows=all_pred[(all_pred.panel==panel)&(all_pred.split!='development_OOF' if fold==0 else ((all_pred.split=='development_OOF')&(all_pred.fold==fold)))].copy()
        joined=rows.merge(raw,on=['dataset','sampleID'],suffixes=('_score',''),validate='many_to_one')
        check('score labels aligned '+file.name,np.array_equal(joined.y_score,joined.y))
        x=joined[features].to_numpy()
        close('actual checkpoint replay '+file.name,pipeline.predict_proba(x)[:,1],rows.Subtype)
        h=(x-scaler.mean_)/scaler.scale_
        for weight,bias in zip(mlp.coefs_[:-1],mlp.intercepts_[:-1]):h=np.maximum(h@weight+bias,0)
        independent=expit(h@mlp.coefs_[-1]+mlp.intercepts_[-1]).ravel()
        close('independent MLP forward '+file.name,independent,rows.Subtype)
    check('native probabilities not calibrated',set(a['panel'] for a in fit_audit)==set(panels))
    check('no fit warnings',not json.loads((OUT/'fit_warnings.json').read_text()))
    for cohort,part in raw.groupby('dataset'):
        lock=json.loads((source/f'heldout_{cohort}.json').read_text())
        check('LOCO heldout excluded '+cohort,cohort not in lock['training_cohorts'] and len(lock['training_cohorts'])==7 and not lock['held_out_labels_used_for_fit'])
        x=part[['MLH1_raw','MSH2_raw','MSH6_raw','PMS2_raw']].to_numpy()
        mean=x.mean(axis=0);sd=x.std(axis=0,ddof=1)
        close('MMR4 archived mean '+cohort,mean,lock['held_out_scaler']['mean'])
        close('MMR4 archived sample SD '+cohort,sd,lock['held_out_scaler']['sample_sd'])
        p=expit((np.asarray(lock['cutoffs'])-(x-mean)/sd)/.5).max(axis=1)
        close('MMR4 independent LOCO replay '+cohort,p,part.MMR4_LOCO)
        j=pred[pred.dataset==cohort].merge(part[['dataset','sampleID','MMR4_LOCO']],on=['dataset','sampleID'],suffixes=('','_source'),validate='many_to_one')
        close('MMR4 identical between models '+cohort,j.MMR4_LOCO,j.MMR4_LOCO_source)
    cols=['AUROC','BA','sensitivity','specificity','Brier']
    for _,row in metrics.iterrows():
        part=pred[(pred.panel==row.panel)&(pred.split==row.split)&(pred.dataset==row.dataset)]
        close('independent metrics '+row.panel+' '+row.split+' '+row.dataset+' '+row.component,metric_values(part.y,part[row.component]),row[cols].to_numpy(dtype=float))
        check('metric sample count '+row.panel+row.split+row.dataset+row.component,len(part)==row.n and int(part.y.sum())==row.MSI)
    for _,row in macro.iterrows():
        part=metrics[(metrics.panel==row.panel)&(metrics.split==row.split)&(metrics.component==row.component)]
        close('macro average '+row.panel+row.split+row.component,part[cols].mean().to_numpy(),row[cols].to_numpy(dtype=float))
    control=pd.read_csv(OUT/'historical10_runtime_comparison.csv')
    check('historical10 control scope',len(control)==1212)
    close('historical10 numerical probabilities',control.Subtype_refit,control.Subtype_historical)
    close('historical10 numerical fusion',control.Fusion05_refit,control.Fusion05_historical)
    archive=np.load(OUT/'bootstrap_indices.npz')
    index_meta=json.loads((OUT/'bootstrap_index_metadata.json').read_text())
    rng=np.random.default_rng(2026)
    for info in index_meta:
        cohort=info['dataset'];part=pred[(pred.panel=='panel7')&(pred.dataset==cohort)&(pred.split!='development_apparent')].sort_values('sampleID')
        check('bootstrap sample order '+cohort,part.sampleID.tolist()==info['sample_order'])
        y=part.y.to_numpy();pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
        idx=archive[cohort]
        expected=np.concatenate([rng.choice(pos,(2000,len(pos)),replace=True),rng.choice(neg,(2000,len(neg)),replace=True)],axis=1)
        check('single seeded RNG index replay '+cohort,np.array_equal(expected,idx))
        check('bootstrap preserves MSI MSS '+cohort,(y[idx[:,:len(pos)]]==1).all() and (y[idx[:,len(pos):]]==0).all())
        # Verify quantiles by independent pair comparisons for all draws on one component per cohort.
        values=np.array([auc(y[index],part.Subtype.to_numpy()[index]) for index in idx])
        row=metrics[(metrics.panel=='panel7')&(metrics.dataset==cohort)&(metrics.split==part.split.iloc[0])&(metrics.component=='Subtype')].iloc[0]
        close('independent bootstrap interval '+cohort,np.quantile(values,[.025,.975]),row[['AUROC_CI_low','AUROC_CI_high']].to_numpy(dtype=float))
    from openpyxl import load_workbook
    wb=load_workbook(OUT/'Bulk_7_vs12_model_scores.xlsx',read_only=True,data_only=True)
    for sheet,path in [('Macro_summary','macro_metrics.csv'),('Cohort_metrics','cohort_metrics.csv'),('Panel_membership','panel_membership.csv'),('Paired_differences','paired_differences.csv')]:
        values=list(wb[sheet].values);frame=pd.read_csv(OUT/path)
        check('workbook headers '+sheet,list(values[0])==list(frame.columns))
        check('workbook rows '+sheet,len(values)-1==len(frame))
        for saved,actual in zip(values[1:],frame.itertuples(index=False,name=None)):
            for a,b in zip(saved,actual):
                if isinstance(b,(int,float,np.number)) and not isinstance(b,(bool,np.bool_)):check('workbook numeric '+sheet,abs(a-b)<1e-12)
                else:check('workbook value '+sheet,a==b)
    for sheet,split in [('Development_OOF_scores','development_OOF'),('Development_apparent','development_apparent'),('Retrospective_scores','retrospective')]:
        check('workbook scores '+sheet,wb[sheet].max_row-1==int((pred.split==split).sum()))
    figure=pd.read_csv(OUT/'figure_values.csv')
    expected=metrics[(metrics.split!='development_apparent')&(metrics.component.isin(['Subtype','Fusion05']))].reset_index(drop=True)
    check('figure row coverage',figure[['panel','split','dataset','component']].equals(expected[['panel','split','dataset','component']]))
    close('figure numeric accuracy',figure.AUROC,expected.AUROC)
    wb.close()
    report={'status':'passed','n_checks':len(checks),'checks':checks,'metric_method':'Independent positive-negative pair comparisons',
            'checkpoint_method':'Actual replay plus independent MLP forward and independent training scaler moments',
            'limitations':'Verification of retained computational inputs, not independent biological measurement or clinical validation'}
    (OUT/'verification.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print('PASS:',len(checks),'checks')

if __name__=='__main__':main()
