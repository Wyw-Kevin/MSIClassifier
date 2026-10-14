"""Two predeclared bulk panels; historical inputs and existing LOCO MMR4.

Run: python run_analysis.py [--output DIRECTORY] [--replay]
Replay uses only the archived source copies beside this script.
"""
from pathlib import Path
import argparse, hashlib, json, platform, random, shutil, warnings
import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, brier_score_loss
import joblib

HERE = Path(__file__).resolve().parent
ARTICLE = HERE.parent
SEED = 2026
DEV = ['GSE39582', 'TCGA-COADREAD']
OLD10 = ['cS27','cTNI16','cM06','cTNI26','cM04','cTNI05','cTNI02','cTNI10','cP1','cTNI13']
SOURCES = {
    'base.csv':'bulk_10panel_mmr4_fixed05_20261005/input_features.csv',
    'fractions.csv':'pybayesprism9_seed2026/subtype_over_major.csv',
    'development.csv':'bulk_selection_bootstrap_seed2026_20261012/development_inputs.csv',
    'screening.csv':'bulk_selection_bootstrap_seed2026_20261012/full_data_selection_replay.csv',
    'ledger.csv':'scRNA_AUROC_threshold_all_candidates_seed2026_20261013/inputs/candidate_ledger.csv',
    'coverage.json':'scRNA_AUROC_threshold_all_candidates_seed2026_20261013/transfer_coverage_audit.json',
    'classifier_protocol.json':'bulk_bayesprism10_dev_strict_exploratory_20261004/protocol.json',
    'mmr4.csv':'bulk_mmr4_loco_fixed05_seed2026_20261012/mmr4_loco_predictions.csv',
    'cutoffs.csv':'bulk_mmr4_loco_fixed05_seed2026_20261012/tables/table_mmr4_loco_cutoffs.csv',
    'mmr4_protocol.json':'bulk_mmr4_loco_fixed05_seed2026_20261012/protocol.json',
    'historical_predictions.csv':'bulk_mmr4_loco_fixed05_seed2026_20261012/predictions_10panel_fixed05.csv',
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')

def csv(df, path):
    df.to_csv(path, index=False, float_format='%.17g')

def align(df):
    assert not df.duplicated(['dataset','sampleID']).any()
    return df.set_index(['dataset','sampleID']).sort_index()

def source_inputs(out, replay):
    archive = out/'source_archive'
    archive.mkdir(exist_ok=True)
    manifest = []
    for name, rel in SOURCES.items():
        src = HERE/'source_archive'/name if replay else ARTICLE/rel
        dest = archive/name
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        manifest.append({'archive':name,'original_path':str(ARTICLE/rel),'sha256':sha(dest)})
    lock_names = sorted((HERE/'source_archive' if replay else ARTICLE/'bulk_mmr4_loco_fixed05_seed2026_20261012/locks').glob('heldout_*.json'))
    assert len(lock_names)==8
    for src in lock_names:
        dest=archive/src.name
        if src.resolve()!=dest.resolve(): shutil.copy2(src,dest)
        manifest.append({'archive':src.name,'original_path':str(ARTICLE/'bulk_mmr4_loco_fixed05_seed2026_20261012/locks'/src.name),'sha256':sha(dest)})
    save_json(out/'source_manifest.json',manifest)
    return archive

def prepare(out, archive):
    base = align(pd.read_csv(archive/'base.csv'))
    fractions=align(pd.read_csv(archive/'fractions.csv'))
    ledger=pd.read_csv(archive/'ledger.csv')
    assert len(ledger)==71 and (ledger['n']>=50).all()
    candidates=ledger.loc[ledger.auc_best_direction>0.70,'feature'].tolist()
    assert len(candidates)==17
    coverage=json.loads((archive/'coverage.json').read_text())
    assert coverage['status']=='passed' and not coverage['unsupported_selected_candidates']
    assert ledger.set_index('feature').loc[candidates,'GSE132465_checkpoint_supported_parent'].all()
    raw=base.drop(columns=['MMR4']).copy()
    canonical=[f for f in OLD10 if f in candidates]+[f for f in candidates if f not in OLD10]
    discrepancies=[]
    for f in OLD10:
        discrepancies.append({'feature':f,'max_absolute_source_difference':float(np.max(np.abs(base[f]-fractions.loc[base.index,f])))})
    assert max(d['max_absolute_source_difference'] for d in discrepancies)<1e-12
    for f in candidates:
        if f not in raw: raw[f]=fractions.loc[base.index,f]
    score=align(pd.read_csv(archive/'mmr4.csv'))
    assert raw.index.equals(score.index)
    assert np.array_equal(raw.y, score.y) and np.array_equal(raw.scope,score.scope)
    raw['MMR4_LOCO']=score.MMR4_LOCO
    assert len(raw)==1212 and raw.scope.value_counts().to_dict()=={'development':892,'external':320}
    assert set(raw[raw.scope=='development'].index.get_level_values(0))==set(DEV)
    assert set(raw[raw.scope=='development'].fold)=={1,2,3,4,5}
    assert (raw[raw.scope=='external'].fold==0).all()
    x=raw[candidates].to_numpy()
    assert np.isfinite(x).all() and (x>=0).all() and (x<=1+1e-12).all()
    dev=raw[raw.scope=='development']
    dev_original=align(pd.read_csv(archive/'development.csv'))
    assert dev.index.equals(dev_original.index) and np.array_equal(dev.y,dev_original.y)
    assert np.allclose(dev[candidates],dev_original[candidates],atol=1e-14,rtol=0)
    replay=pd.read_csv(archive/'screening.csv').set_index('feature')
    directions=ledger.set_index('feature').direction_sign
    panel_records=[]
    for f in candidates:
        record={'feature':f,'parent':ledger.set_index('feature').loc[f,'group_class'],
                'scRNA_direction':int(directions[f]),'scRNA_oriented_AUROC':float(ledger.set_index('feature').loc[f,'auc_best_direction'])}
        for cohort in DEV:
            part=dev.xs(cohort)
            record[cohort+'_oriented_AUROC']=float(roc_auc_score(part.y, part[f]*directions[f]))
        assert abs(record['GSE39582_oriented_AUROC']-replay.loc[f,'GSE39582_oriented_AUROC'])<1e-12
        assert abs(record['TCGA-COADREAD_oriented_AUROC']-replay.loc[f,'TCGA_oriented_AUROC'])<1e-12
        record['panel7']=all(record[c+'_oriented_AUROC']>0.60 for c in DEV)
        record['panel12']=all(record[c+'_oriented_AUROC']>0.50 for c in DEV)
        panel_records.append(record)
    table=pd.DataFrame(panel_records)
    panels={name:[f for f in canonical if bool(table.set_index('feature').loc[f,name])]
            for name in ['panel7','panel12']}
    assert len(panels['panel7'])==7 and len(panels['panel12'])==12
    assert set(panels['panel7'])<=set(panels['panel12'])
    panels['historical10_control']=OLD10
    raw=raw.reset_index()
    csv(raw,out/'inputs/all_cohort_features.csv')
    csv(table,out/'panel_membership.csv')
    save_json(out/'panels_locked_before_fit.json',panels)
    save_json(out/'input_audit.json',{'status':'passed','n_samples':len(raw),'candidate_n':len(candidates),
              'finite_nonnegative':True,'label_fold_alignment':True,'source_fraction_differences':discrepancies,
              'preserved_exact_historical10_columns':True,'checkpoint_coverage_source':'coverage.json'})
    return raw,panels

def fit_panels(raw,panels,params,out):
    dev=raw[raw.scope=='development'].copy().reset_index(drop=True)
    ext=raw[raw.scope=='external'].copy().reset_index(drop=True)
    records=[]; audit=[]; captured=[]
    for panel,features in panels.items():
        for fold in list(range(1,6))+[0]:
            training=dev if fold==0 else dev[dev.fold!=fold]
            subsets=[('development_apparent',dev),('retrospective',ext)] if fold==0 else [('development_OOF',dev[dev.fold==fold])]
            model=Pipeline([('scaler',StandardScaler()),('mlp',MLPClassifier(**params))])
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter('always')
                model.fit(training[features].to_numpy(),training.y.to_numpy())
            captured += [{'panel':panel,'fold':fold,'warning':str(w.message)} for w in ws]
            artifact={'pipeline':model,'features':features,'seed':SEED,'fold':fold,
                      'training_keys':training[['dataset','sampleID']].to_dict('records'),
                      'classifier_parameters':params,'panel':panel}
            file=out/'models'/f'{panel}_fold{fold}.joblib'
            joblib.dump(artifact,file)
            audit.append({'panel':panel,'fold':fold,'n_train':len(training),'n_positive':int(training.y.sum()),
                          'n_iter':int(model.named_steps['mlp'].n_iter_),'final_loss':float(model.named_steps['mlp'].loss_),
                          'checkpoint':str(file.relative_to(out)),'checkpoint_sha256':sha(file)})
            for split,part in subsets:
                pred=part[['dataset','sampleID','y','scope','fold','MMR4_LOCO']].copy()
                pred['panel']=panel;pred['n_features']=len(features);pred['split']=split
                pred['Subtype']=model.predict_proba(part[features].to_numpy())[:,1]
                pred['Fusion05']=0.5*pred.Subtype+0.5*pred.MMR4_LOCO
                pred['Subtype_class05']=(pred.Subtype>=.5).astype(int)
                pred['Fusion05_class05']=(pred.Fusion05>=.5).astype(int)
                records.append(pred)
            print(f'Fitted {panel} fold {fold}, n={len(training)}, iterations={audit[-1]["n_iter"]}',flush=True)
    save_json(out/'fit_audit.json',audit)
    save_json(out/'fit_warnings.json',captured)
    df=pd.concat(records,ignore_index=True).sort_values(['panel','split','dataset','sampleID']).reset_index(drop=True)
    control=df[df.panel=='historical10_control'].copy()
    csv(control,out/'historical10_runtime_predictions.csv')
    df=df[df.panel!='historical10_control'].copy().reset_index(drop=True)
    assert len(df)==4208
    csv(df,out/'predictions.csv')
    return df,control

def evaluate(pred,out):
    rows=[]
    for (panel,split,dataset),part in pred.groupby(['panel','split','dataset'],sort=True):
        for component in ['Subtype','MMR4_LOCO','Fusion05']:
            y=part.y.to_numpy();p=part[component].to_numpy();positive=p>=.5
            rows.append({'panel':panel,'n_features':int(part.n_features.iloc[0]),'split':split,'dataset':dataset,
                         'component':component,'n':len(y),'MSI':int(y.sum()),'MSS':int((1-y).sum()),
                         'AUROC':roc_auc_score(y,p),'BA':balanced_accuracy_score(y,positive),
                         'sensitivity':float(positive[y==1].mean()),'specificity':float((~positive[y==0]).mean()),
                         'Brier':brier_score_loss(y,p)})
    metrics=pd.DataFrame(rows)
    macro=metrics.groupby(['panel','split','component'],sort=True)[['AUROC','BA','sensitivity','specificity','Brier']].mean().reset_index()
    macro['n_cohorts']=macro['split'].map({'development_OOF':2,'development_apparent':2,'retrospective':6})
    csv(metrics,out/'cohort_metrics.csv');csv(macro,out/'macro_metrics.csv')
    return metrics,macro

def bootstrap(pred,metrics,macro,out):
    rng=np.random.default_rng(SEED)
    reps=2000;arrays={};boot={};index_metadata=[]
    for cohort in sorted(pred.dataset.unique()):
        identity=pred[(pred.panel=='panel7')&(pred.dataset==cohort)&(pred.split!='development_apparent')].sort_values('sampleID')
        y=identity.y.to_numpy();pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
        # One stream advances across all cohorts and resamples; no seed offsets.
        index=np.concatenate([rng.choice(pos,(reps,len(pos)),replace=True),rng.choice(neg,(reps,len(neg)),replace=True)],axis=1)
        arrays[cohort]=index.astype('int32')
        index_metadata.append({'dataset':cohort,'sample_order':identity.sampleID.tolist(),'n_MSI':len(pos),'n_MSS':len(neg)})
        for split in sorted(pred[pred.dataset==cohort].split.unique()):
            for panel in ['panel7','panel12']:
                part=pred[(pred.dataset==cohort)&(pred.split==split)&(pred.panel==panel)].sort_values('sampleID')
                assert np.array_equal(part.sampleID,identity.sampleID) and np.array_equal(part.y,y)
                for component in ['Subtype','MMR4_LOCO','Fusion05']:
                    ranks=rankdata(part[component].to_numpy()[index],axis=1,method='average')
                    boot[(panel,split,cohort,component)]=(ranks[:,:len(pos)].sum(axis=1)-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))
    np.savez_compressed(out/'bootstrap_indices.npz',**arrays)
    save_json(out/'bootstrap_index_metadata.json',index_metadata)
    for i,row in metrics.iterrows():
        lo,hi=np.quantile(boot[(row.panel,row.split,row.dataset,row.component)],[.025,.975])
        metrics.loc[i,'AUROC_CI_low']=lo;metrics.loc[i,'AUROC_CI_high']=hi
    differences=[]
    for i,row in macro.iterrows():
        cohorts=DEV if row.split.startswith('development') else sorted(set(pred.dataset)-set(DEV))
        values=np.mean([boot[(row.panel,row.split,c,row.component)] for c in cohorts],axis=0)
        lo,hi=np.quantile(values,[.025,.975]);macro.loc[i,'AUROC_CI_low']=lo;macro.loc[i,'AUROC_CI_high']=hi
        if row.panel=='panel12':
            seven=np.mean([boot[('panel7',row.split,c,row.component)] for c in cohorts],axis=0)
            lo,hi=np.quantile(values-seven,[.025,.975])
            observed=row.AUROC-float(macro[(macro.panel=='panel7')&(macro.split==row.split)&(macro.component==row.component)].AUROC.iloc[0])
            differences.append({'scope':'macro','split':row.split,'dataset':'equal-weight cohort macro','component':row.component,
                                'AUROC_delta_12_minus_7':observed,'CI_low':lo,'CI_high':hi})
    for (split,cohort,component),part in metrics.groupby(['split','dataset','component']):
        values=boot[('panel12',split,cohort,component)]-boot[('panel7',split,cohort,component)]
        lo,hi=np.quantile(values,[.025,.975]);a=part.set_index('panel').AUROC
        differences.append({'scope':'cohort','split':split,'dataset':cohort,'component':component,
                            'AUROC_delta_12_minus_7':float(a['panel12']-a['panel7']),'CI_low':lo,'CI_high':hi})
    differences=pd.DataFrame(differences)
    csv(metrics,out/'cohort_metrics.csv');csv(macro,out/'macro_metrics.csv');csv(differences,out/'paired_differences.csv')
    save_json(out/'bootstrap_protocol.json',{'seed':SEED,'resamples':reps,'unit':'sample','stratification':'within cohort, MSI/MSS',
              'paired_between_panels':True,'single_advancing_rng_stream':True,'interval':'percentile 95%',
              'fixed_predictions':True,'model_or_feature_selection_refitted':False,
              'GEO_patient_clustering_independently_audited':False,'scope':'Conditional discrimination uncertainty only; not full development uncertainty'})
    return metrics,macro,differences

def products(out,pred,metrics,macro,differences):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    notes=pd.DataFrame([
        ['Model','StandardScaler + MLP, 4 ReLU units, alpha=1, seed=2026; native predict_proba'],
        ['Scores','Higher score indicates MSI; Fusion05=0.5*Subtype+0.5*MMR4_LOCO; class threshold=0.5'],
        ['Panel selection','scRNA-oriented AUROC >0.70 yields 17; both development AUROCs >0.60 / >0.50 yield 7 / 12'],
        ['OOF','Archived five development folds; panel selected on full development labels before classifier CV'],
        ['Input identity','Historical BayesPrism subtype-over-parent fractions; no new deconvolution or DECODE identity'],
        ['Evaluation','Six previously explored GEO cohorts: retrospective evaluation'],
        ['MMR4','Eight-cohort LOCO cutoffs; held-out labels excluded per round; other cohorts enter fitting'],
        ['Intervals','2000 paired, outcome-stratified, sample-level resamples of fixed predictions; GEO patient clustering not audited'],
        ['Macro','Arithmetic mean of cohort AUROCs; not AUROC of pooled samples'],
        ['Apparent','Full model predictions on the development fitting samples; descriptive only'],
        ['Preservation','No manuscript, primary classifier or historical weights overwritten'],
    ],columns=['Item','Definition'])
    file=out/'Bulk_7_vs12_model_scores.xlsx'
    with pd.ExcelWriter(file,engine='openpyxl') as writer:
        for name,frame in [('Notes',notes),('Macro_summary',macro),('Cohort_metrics',metrics),('Panel_membership',pd.read_csv(out/'panel_membership.csv')),
                           ('Paired_differences',differences),('Development_OOF_scores',pred[pred.split=='development_OOF']),
                           ('Development_apparent',pred[pred.split=='development_apparent']),('Retrospective_scores',pred[pred.split=='retrospective'])]:
            frame.to_excel(writer,sheet_name=name,index=False)
        for sheet in writer.book:
            sheet.freeze_panes='A2';sheet.auto_filter.ref=sheet.dimensions
            for cells in sheet.columns:
                name=cells[0].value
                sheet.column_dimensions[cells[0].column_letter].width=min(44,max(15,len(str(name))+3))
            for row in sheet.iter_rows(min_row=2):
                for cell in row:
                    if isinstance(cell.value,float):cell.number_format='0.0000'
    cohorts=DEV+sorted(set(pred.dataset)-set(DEV))
    plotted=metrics[(metrics.split!='development_apparent')&(metrics.component.isin(['Subtype','Fusion05']))].copy()
    csv(plotted,out/'figure_values.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4.7),sharey=True,constrained_layout=True)
    for ax,component,title in zip(axes,['Subtype','Fusion05'],['Subtype model','Fixed 0.5:0.5 fusion with LOCO MMR4']):
        for panel,offset,color in [('panel7',-.1,'#2678B2'),('panel12',.1,'#D95F02')]:
            subset=plotted[(plotted.component==component)&(plotted.panel==panel)].set_index('dataset').loc[cohorts]
            ax.plot(subset.AUROC.to_numpy(),np.arange(8)+offset,'o',color=color,label=panel.replace('panel','')+' subtypes',markersize=6)
        ax.axhline(1.5,color='.75',lw=.8,ls='--');ax.set_xlim(.45,1.015);ax.set_xlabel('AUROC');ax.set_title(title,fontsize=11)
        ax.grid(axis='x',alpha=.2);ax.set_yticks(np.arange(8),[c+(' (OOF)' if c in DEV else '') for c in cohorts])
        ax.legend(loc='lower left',fontsize=9)
    axes[0].invert_yaxis()
    fig.savefig(out/'Bulk_7_vs12_AUROC.png',dpi=220);fig.savefig(out/'Bulk_7_vs12_AUROC.pdf');plt.close(fig)
    text='''# Bulk 7 versus 12 subtype models

The Excel workbook contains individual MSI-oriented prediction scores, development five-fold OOF scores, full-fit development apparent scores, and six-cohort retrospective evaluation. Cohort AUROC, balanced accuracy at 0.5, sensitivity, specificity and Brier score are provided for subtype-only, existing LOCO MMR4 and fixed equal-weight fusion.

This is a predeclared downstream comparison using historical BayesPrism subtype/parent inputs, the existing five folds and the current four-unit MLP settings. Selection is restricted to the 17 candidates with source scRNA oriented AUROC >0.70. The 7/12 panels require both development-oriented AUROCs >0.60/>0.50; source direction is frozen. Input order is the historical ten-feature order restricted to each panel, followed by additional candidates in ledger order. No retrospective outcomes select panels, hyperparameters or fusion weights.

Classifier scalers use training rows only, within each fold. Full-development feature screening and cohort-adaptive measurements remain outside CV. MMR4 uses the archived eight-cohort leave-one-cohort-out thresholds and per-cohort mean/sample-SD normalization, temperature 0.5. It is not end-to-end LOCO, and other retrospective cohorts enter each cutoff fit. These cohorts were previously viewed.

The 95% intervals use 2,000 paired MSI/MSS-stratified sample resamples of fixed scores. The RNG is seeded 2026 once and advances through the resampling; the actual indices are retained. These intervals do not include feature-selection, classifier-fitting or preprocessing uncertainty. GEO patient clustering has not been independently audited.

Source copies and SHA256 hashes, all 18 actual checkpoints (5 folds plus full fit for each 7/12 panel and a historical10 numerical control), independent verification, and plotted values are retained. No main manuscript or existing model is replaced.

Reproduce using Python 3.12.12, NumPy 1.26.4, pandas 2.3.3, SciPy 1.16.3 and scikit-learn 1.7.2:

    PYTHONHASHSEED=2026 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run_analysis.py --replay --output /tmp/bulk7_12_replay
    python verify_analysis.py /tmp/bulk7_12_replay

All English workbook tables are machine-readable; the main figure summarizes per-cohort subtype and fusion AUROCs.
'''
    (out/'README.md').write_text(text)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=HERE);parser.add_argument('--replay',action='store_true')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    for name in ['inputs','models']: (out/name).mkdir(exist_ok=True)
    random.seed(SEED);np.random.seed(SEED)
    archive=source_inputs(out,args.replay)
    raw,panels=prepare(out,archive)
    params=json.loads((archive/'classifier_protocol.json').read_text())['models']['mlp_4_l2']
    params['hidden_layer_sizes']=tuple(params['hidden_layer_sizes'])
    assert params['random_state']==SEED and params['hidden_layer_sizes']==(4,) and params['alpha']==1.0
    protocol={'status':'Predeclared exploratory bulk panel comparison; manuscript unchanged','seed':SEED,
              'panels':panels,'candidate_filter':'Source scRNA oriented AUROC strictly >0.70; 17 candidates',
              'panel7_filter':'Both development oriented AUROCs strictly >0.60',
              'panel12_filter':'Both development oriented AUROCs strictly >0.50',
              'direction':'Frozen source scRNA direction, not bulk-optimized',
              'feature_order':'Historical10 restricted to candidates, followed by other candidates in source ledger order',
              'classifier':params,'folds':'Archived five folds, unchanged','scaling':'Development training rows only, within each fold',
              'input_identity':'Historical Python BayesPrism71 subtype/parent fractions; unchanged cohort-adaptive measurements',
              'MMR4':'Archived eight-cohort LOCO; gene-order max sigmoid; target-cohort unsupervised normalization',
              'fusion_weights':[.5,.5],'class_threshold':.5,'evaluation':'Six previously explored cohorts; retrospective',
              'CV_scope':'Fixed-panel downstream classifier CV; selection and preprocessing not nested; MMR4 not classifier-fold isolated',
              'software':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__}}
    save_json(out/'protocol.json',protocol)
    pred,control=fit_panels(raw,panels,params,out)
    historical=pd.read_csv(archive/'historical_predictions.csv');historical=historical[historical.model=='mlp_4_l2']
    control=control[control.split!='development_apparent'].merge(historical[['dataset','sampleID','Subtype','Fusion05']],on=['dataset','sampleID'],validate='one_to_one',suffixes=('_refit','_historical'))
    diff=float(np.max(np.abs(control.Subtype_refit-control.Subtype_historical)))
    assert len(control)==1212 and diff<1e-12
    csv(control,out/'historical10_runtime_comparison.csv')
    save_json(out/'historical10_control.json',{'status':'passed','n':len(control),'max_absolute_probability_difference':diff})
    metrics,macro=evaluate(pred,out);metrics,macro,paired=bootstrap(pred,metrics,macro,out)
    products(out,pred,metrics,macro,paired)
    print(macro[['panel','split','component','AUROC','BA']].to_string(index=False),flush=True)

if __name__=='__main__': main()
