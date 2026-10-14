"""New LOCO branch; archived subtype scores and all source artifacts are preserved."""
from pathlib import Path
import hashlib
import itertools
import json
import platform
import shutil
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import roc_auc_score,balanced_accuracy_score
from loco import GENES,loco_predictions,fuse_scores

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'bulk_10panel_mmr4_fixed05_20261005'
DEV=['GSE39582','TCGA-COADREAD']
EXT=['GSE24550','GSE25071','GSE27544','GSE35896','GSE39084','GSE75316']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save_json(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)))

def main():
    assert not (HERE/'complete.json').exists(),'Preserve completed run; use a new directory'
    (HERE/'tables').mkdir(exist_ok=True);(HERE/'locks').mkdir(exist_ok=True)
    sources=[SOURCE/n for n in ['input_features.csv','predictions_10panel_fixed05.csv','protocol.json','MMR4_lock.json','feature_order.csv']]
    hashes={str(p):sha(p) for p in sources}
    frozen={'seed':2026,'status':'User-approved cohort-adaptive MMR4 LOCO, conditional fixed-subtype fusion',
        'cohorts':sorted(DEV+EXT),'n_samples':1212,'genes':GENES,'temperature':.5,
        'normalization':'Independent within-cohort mean and sample SD (ddof=1); held-out expression but no held-out labels',
        'cutoff_grid':'[-4,0] inclusive, step 0.5, Cartesian four-gene product (6561)',
        'fit_objective':'Equal-weight mean AUROC over seven training cohorts',
        'tie_rule':'Training macro AUROC rounded12; training macro BA at 0.5 rounded12; gene-order lexicographic',
        'historical_threshold_anchor_used':False,'subtype_refitted':False,'fusion_weights':[.5,.5],
        'classification_threshold':.5,'source_sha256':hashes,
        'limitations':['Previously explored cohorts; no prospectively untouched external validation',
            'Subtype panel, deconvolution and classifier predictions remain fixed; this is not end-to-end LOCO',
            'MMR4 uses target-cohort unsupervised normalization; no direct frozen single-sample inference'],
        'software':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__}}
    save_json(HERE/'protocol.json',frozen)
    for p in sources:shutil.copy2(p,HERE/('source_'+p.name))
    data=pd.read_csv(SOURCE/'input_features.csv',float_precision='round_trip').rename(columns={'MMR4':'MMR4_legacy_all8'})
    assert len(data)==1212 and set(data.dataset)==set(DEV+EXT)
    assert not data.duplicated(['dataset','sampleID']).any()
    grid=np.array(list(itertools.product(np.arange(-4,.01,.5),repeat=4)))
    predicted,locks=loco_predictions(data,grid)
    threshold_rows=[];scalers=[]
    for ds,lock in locks.items():
        scan=lock.pop('training_grid');scan.to_csv(HERE/'locks'/('training_grid_'+ds+'.csv'),index=False)
        save_json(HERE/'locks'/('heldout_'+ds+'.json'),lock)
        threshold_rows.append(dict(held_out_cohort=ds,training_cohorts=';'.join(lock['training_cohorts']),
            **dict(zip(GENES,lock['cutoffs'])),training_macro_AUROC=lock['training_macro_AUROC'],
            training_macro_BA=lock['training_macro_BA'],temperature=.5,held_out_labels_used_for_fit=False))
        for j,g in enumerate(GENES):scalers.append({'dataset':ds,'gene':g,'mean_expression':lock['held_out_scaler']['mean'][j],
            'sample_sd_expression':lock['held_out_scaler']['sample_sd'][j],'normalization_uses_labels':False})
    pd.DataFrame(threshold_rows).to_csv(HERE/'tables/table_mmr4_loco_cutoffs.csv',index=False)
    pd.DataFrame(scalers).to_csv(HERE/'tables/table_mmr4_scalers.csv',index=False)
    predicted['MMR4']=predicted.MMR4_LOCO
    predicted.to_csv(HERE/'input_features.csv',index=False)
    predicted[['dataset','sampleID','y','scope','MMR4_LOCO','MMR4_legacy_all8']].to_csv(HERE/'mmr4_loco_predictions.csv',index=False)
    pred=pd.read_csv(SOURCE/'predictions_10panel_fixed05.csv',float_precision='round_trip')
    old_subtype=pred.Subtype.to_numpy().copy()
    assert len(pred)==1212*11 and pred.model.nunique()==11 and pred.n_features.eq(10).all()
    pred=pred.rename(columns={'MMR4':'MMR4_legacy_all8','Fusion05':'Fusion05_legacy_all8'})
    lookup=predicted.set_index(['dataset','sampleID']).MMR4_LOCO
    pred['MMR4']=lookup.reindex(pd.MultiIndex.from_frame(pred[['dataset','sampleID']])).to_numpy()
    pred['MMR4_LOCO']=pred.MMR4
    pred['Fusion05']=fuse_scores(pred.Subtype,pred.MMR4_LOCO)
    np.testing.assert_array_equal(pred.Subtype,old_subtype)
    assert np.isfinite(pred.Fusion05).all() and not pred.duplicated(['model','dataset','sampleID']).any()
    pred.to_csv(HERE/'predictions_10panel_fixed05.csv',index=False)
    rows=[]
    for (model,ds),p in pred.groupby(['model','dataset'],sort=True):
        row=dict(model=model,dataset=ds,scope=p.scope.iloc[0],n=len(p),n_MSI=int(p.y.sum()))
        for name,col in [('subtype','Subtype'),('MMR4','MMR4'),('fusion_05','Fusion05')]:
            row[name+'_AUROC']=float(roc_auc_score(p.y,p[col]));row[name+'_BA']=float(balanced_accuracy_score(p.y,p[col]>=.5))
        rows.append(row)
    metrics=pd.DataFrame(rows);metrics.to_csv(HERE/'tables/table_11model_subtype_mmr4_fusion.csv',index=False)
    first=metrics.model.iloc[0]
    mmr=metrics[metrics.model==first][['scope','dataset','n','n_MSI','MMR4_AUROC','MMR4_BA']]
    mmr.to_csv(HERE/'tables/table_mmr4_by_cohort.csv',index=False)
    macro=[]
    for subset,part in [('development',metrics[metrics.scope=='development']),('external',metrics[metrics.scope=='external']),('all',metrics)]:
        for model,p in part.groupby('model',sort=True):
            macro.append(dict(scope=subset,model=model,n_cohorts=len(p),**{c:float(p[c].mean()) for c in ['subtype_AUROC','MMR4_AUROC','fusion_05_AUROC','subtype_BA','MMR4_BA','fusion_05_BA']}))
    summary=pd.DataFrame(macro);summary.to_csv(HERE/'tables/table_11model_macro_summary.csv',index=False)
    old_metrics=pd.read_csv(SOURCE/'tables/table_11model_subtype_mmr4_fusion.csv')
    paired=metrics.merge(old_metrics,on=['model','dataset','scope'],suffixes=('_LOCO','_legacy'),validate='one_to_one')
    np.testing.assert_allclose(paired.subtype_AUROC_LOCO,paired.subtype_AUROC_legacy,atol=1e-12)
    paired.to_csv(HERE/'tables/paired_LOCO_vs_legacy.csv',index=False)
    mmr_summary={scope:float(p.MMR4_AUROC.mean()) for scope,p in mmr.groupby('scope')}
    mmr_summary['all']=float(mmr.MMR4_AUROC.mean())
    ranges={scope:{'fusion_min':float(p.fusion_05_AUROC.min()),'fusion_max':float(p.fusion_05_AUROC.max()),
        'subtype_min':float(p.subtype_AUROC.min()),'subtype_max':float(p.subtype_AUROC.max())} for scope,p in summary.groupby('scope')}
    assert hashes=={str(p):sha(p) for p in sources}
    result={'status':'computed; independent verification pending','MMR4_macro_AUROC':mmr_summary,'classifier_ranges':ranges,
        'subtype_scores_unchanged':True,'source_files_unchanged':True,'n_samples':1212,'n_prediction_rows':len(pred)}
    save_json(HERE/'computed_summary.json',result)
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
