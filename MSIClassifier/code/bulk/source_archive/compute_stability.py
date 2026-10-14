"""Recalculate selection stability for the adopted17-candidate/12-feature panel.

Retain the original2000 development bootstrap samples, with full seed replay.
No external outcome enters selection or resampling.
"""
from pathlib import Path
import json, hashlib, shutil
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE=Path(__file__).resolve().parent
ARTICLE=HERE.parent
MODEL=ARTICLE/'bulk_7_vs12_mlp_mmr4_loco_seed2026_20261013'
OLD=ARTICLE/'bulk_selection_bootstrap_seed2026_20261012'
DEV=['GSE39582','TCGA-COADREAD']
SEED=2026;REPS=2000
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()

def dump(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def weighted_auc(x,y,counts):
    order=np.argsort(x,kind='stable');x=x[order];y=y[order];w=counts[order]
    starts=np.r_[0,np.flatnonzero(x[1:]!=x[:-1])+1]
    positive=np.add.reduceat(w*(y==1),starts);negative=np.add.reduceat(w*(y==0),starts)
    return float(np.sum(positive*(np.cumsum(negative)-negative+.5*negative))/(positive.sum()*negative.sum()))

def main():
    out=HERE/'stability';out.mkdir(exist_ok=True)
    sources={'model_input':MODEL/'inputs/all_cohort_features.csv',
             'membership':MODEL/'panel_membership.csv','model_verification':MODEL/'verification.json',
             'model_protocol':MODEL/'protocol.json','model_predictions':MODEL/'predictions.csv',
             'old_indices':OLD/'resampling_records.npz','old_development':OLD/'development_inputs.csv',
             'old_directions':OLD/'candidate_directions.csv','old_verification':OLD/'verification.json'}
    hashes={str(f):sha(f) for f in sources.values()}
    assert json.loads(sources['model_verification'].read_text())['status']=='passed'
    ledger=pd.read_csv(sources['membership']);features=ledger.feature.tolist();direction=ledger.scRNA_direction.to_numpy()
    assert len(features)==17
    panel=ledger.panel12.to_numpy(dtype=bool);assert panel.sum()==12
    raw=pd.read_csv(sources['model_input']);dev=raw[raw.scope=='development'].copy().sort_values(['dataset','sampleID'])
    old_dev=pd.read_csv(sources['old_development']).sort_values(['dataset','sampleID'])
    assert len(dev)==892 and int(dev.y.sum())==129
    assert np.array_equal(dev[['dataset','sampleID','y']],old_dev[['dataset','sampleID','y']])
    parts=[dev[dev.dataset==c].reset_index(drop=True) for c in DEV]
    records=np.load(sources['old_indices'])
    old_features=pd.read_csv(sources['old_directions']).feature.tolist()
    arrays=[g[features].to_numpy()*direction for g in parts]
    labels=[g.y.to_numpy() for g in parts]
    selected=np.zeros((REPS,len(features)),dtype=bool);aucs=np.empty((REPS,2,len(features)))
    rng=np.random.default_rng(SEED)
    checks=0
    for rep in range(REPS):
        for k,c in enumerate(DEV):
            y=labels[k]
            expected=np.concatenate([rng.choice(np.flatnonzero(y==cls),size=int((y==cls).sum()),replace=True) for cls in (0,1)])
            idx=records['indices_'+c][rep]
            assert np.array_equal(expected,idx);checks+=1
            assert int(y[idx].sum())==int(y.sum());checks+=1
            counts=np.bincount(idx,minlength=len(y))
            aucs[rep,k]=[weighted_auc(arrays[k][:,j],y,counts) for j in range(len(features))]
            prior=records['aucs'][rep,k,[old_features.index(f) for f in features]]
            assert np.max(np.abs(prior-aucs[rep,k]))<1e-12;checks+=1
            if rep in [0,499,999,1499,1999]:
                independent=np.array([roc_auc_score(y[idx],arrays[k][idx,j]) for j in range(len(features))])
                assert np.max(np.abs(independent-aucs[rep,k]))<1e-12;checks+=len(features)
        selected[rep]=(aucs[rep]>.5).all(axis=0)
        if (rep+1)%500==0:print('Verified selection resamples',rep+1,flush=True)
    original=np.array([[roc_auc_score(g.y,g[f]*s) for f,s in zip(features,direction)] for g in parts])
    assert np.array_equal((original>.5).all(axis=0),panel)
    assert np.max(np.abs(original[0]-ledger.GSE39582_oriented_AUROC))<1e-12
    assert np.max(np.abs(original[1]-ledger['TCGA-COADREAD_oriented_AUROC']))<1e-12
    table=ledger.rename(columns={'panel7':'historical_comparison_panel7','panel12':'retained_bulk12'}).copy()
    table['bootstrap_selected_count']=selected.sum(axis=0)
    table['bootstrap_inclusion_frequency']=selected.mean(axis=0)
    for k,c in enumerate(DEV):
        quantiles=np.quantile(aucs[:,k,:],[.025,.5,.975],axis=0)
        for q,label in enumerate(['p025','median','p975']):table[c+'_AUROC_'+label]=quantiles[q]
        table[c+'_direction_retained_frequency']=(aucs[:,k,:]>.5).mean(axis=0)
    count=selected.sum(axis=1);common=(selected&panel).sum(axis=1);union=(selected|panel).sum(axis=1)
    overlap=common/union;exact=np.all(selected==panel,axis=1)
    replicate=pd.DataFrame({'replicate':np.arange(1,REPS+1),'selected_count':count,'retained_panel12_count':common,
                            'extra_count':count-common,'Jaccard_with_panel12':overlap,'exact_panel12':exact})
    values,counts=np.unique(count,return_counts=True)
    size=pd.DataFrame({'selected_count':values,'replicates':counts,'frequency':counts/REPS})
    retained=table[table.retained_bulk12]
    summary={'status':'complete; independently verified','seed':SEED,'replicates':REPS,'candidate_count':17,'panel_size':12,
             'n_development':892,'n_MSI':129,'n_MSS':763,'criterion':'Frozen source direction; both development AUROCs strictly >0.50',
             'inclusion_min':float(retained.bootstrap_inclusion_frequency.min()),'inclusion_max':float(retained.bootstrap_inclusion_frequency.max()),
             'selected_count_median':float(np.median(count)),'selected_count_p025':float(np.quantile(count,.025)),
             'selected_count_p975':float(np.quantile(count,.975)),'Jaccard_median':float(np.median(overlap)),
             'Jaccard_p025':float(np.quantile(overlap,.025)),'Jaccard_p975':float(np.quantile(overlap,.975)),
             'exact_panel_recovery':float(exact.mean()),'n_exact_panel_recovery':int(exact.sum()),
             'size12_frequency':float((count==12).mean()),'n_size12':int((count==12).sum()),
             'interpretation':'Conditional selection sensitivity; source screen and cohort-adaptive measurements fixed',
             'adoption':'User selected12 after7/12 comparison was viewed; not prospectively specified',
             'GEO_patient_clustering_independently_audited':False,'verification_checks':checks,
             'frequencies':dict(zip(retained.feature,retained.bootstrap_inclusion_frequency))}
    for name,frame in [('S14.csv',table),('S14_replicate_overlap.csv',replicate),('S14_selected_count_distribution.csv',size),('development_inputs.csv',dev[['dataset','sampleID','y']+features])]:
        frame.to_csv(out/name,index=False,float_format='%.17g')
    np.savez_compressed(out/'resampling_records.npz',aucs=aucs,selected=selected,
                        **{'indices_'+c:records['indices_'+c] for c in DEV})
    dump(out/'summary.json',summary)
    dump(out/'protocol.json',{'seed':SEED,'replicates':REPS,'sampling':'Fixed original cohort-by-outcome strata with replacement',
                             'candidate_n':17,'threshold':.5,'directions':'GSE178341 frozen',
                             'indices_source':'Archived seed2026 samples; exactly replayed with one advancing RNG stream',
                             'external_labels_used':False,'classifier_refitted_in_selection_resamples':False,'source_hashes':hashes})
    assert all(sha(Path(f))==h for f,h in hashes.items())
    dump(out/'verification.json',{'status':'passed','checks':checks,'indices_all_replayed':True,
                                  'all_oriented_AUROCs_match_verified_archived_resamples':True,
                                  'independent_sklearn_AUROC_checks':170,'model_input_frozen':True,'original_sources_unchanged':True})
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
