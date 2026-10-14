"""Training-only MMR4 optimization and cohort-adaptive LOCO scores."""
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata

GENES=['MLH1','MSH2','MSH6','PMS2']

def fit_cutoffs(training,grid):
    """Only training-cohort labels enter this optimizer, including tie handling."""
    cohorts=sorted(training.dataset.unique())
    z=training[GENES].to_numpy(dtype=float)
    y=training.y.to_numpy(dtype=int)
    grid=np.asarray(grid,dtype=float)
    if grid.ndim!=2 or grid.shape[1]!=4 or not np.isfinite(grid).all():
        raise ValueError('Invalid four-gene cutoff grid')
    if not np.isfinite(z).all() or not set(y)<=set([0,1]):
        raise ValueError('Invalid standardized training expression or labels')
    masks=[training.dataset.eq(ds).to_numpy() for ds in cohorts]
    for mask in masks:
        if len(np.unique(y[mask]))!=2:raise ValueError('Each training cohort requires two outcome classes')
    auc_blocks=[];ba_blocks=[]
    for start in range(0,len(grid),256):
        scores=expit((grid[start:start+256,None,:]-z[None,:,:])/.5).max(axis=2)
        aucs=[];bas=[]
        for mask in masks:
            target=y[mask];s=scores[:,mask];positive=target==1;negative=target==0
            ranks=rankdata(s,axis=1,method='average');np_=positive.sum();nn_=negative.sum()
            aucs.append((ranks[:,positive].sum(axis=1)-np_*(np_+1)/2)/(np_*nn_))
            bas.append(.5*((s[:,positive]>=.5).mean(axis=1)+(s[:,negative]<.5).mean(axis=1)))
        auc_blocks.append(np.stack(aucs,axis=1));ba_blocks.append(np.stack(bas,axis=1))
    av=np.concatenate(auc_blocks);bv=np.concatenate(ba_blocks)
    scan=pd.DataFrame(grid,columns=GENES)
    for j,ds in enumerate(cohorts):
        scan[ds+'_AUROC']=av[:,j];scan[ds+'_BA']=bv[:,j]
    scan['training_macro_AUROC']=av.mean(axis=1);scan['training_macro_BA']=bv.mean(axis=1)
    ranked=scan.assign(rank_auc=scan.training_macro_AUROC.round(12),rank_ba=scan.training_macro_BA.round(12))
    best=ranked.sort_values(['rank_auc','rank_ba']+GENES,ascending=[False,False]+[True]*4).iloc[0]
    return {'cutoffs':best[GENES].to_numpy(dtype=float),'training_cohorts':cohorts,
            'training_macro_AUROC':float(best.training_macro_AUROC),
            'training_macro_BA':float(best.training_macro_BA),'training_grid':scan}

def loco_predictions(data,grid):
    """Fit on each complement, then score its omitted cohort without target labels."""
    data=data.copy().reset_index(drop=True)
    if data.duplicated(['dataset','sampleID']).any():raise ValueError('Duplicate cohort/sample keys')
    raw=[g+'_raw' for g in GENES]
    if not np.isfinite(data[raw].to_numpy(dtype=float)).all():raise ValueError('Nonfinite gene input')
    scaled=data.copy();scalers={}
    for ds,part in data.groupby('dataset',sort=True):
        expression=part[raw];mu=expression.mean().to_numpy();sd=expression.std(ddof=1).to_numpy()
        if not np.isfinite(sd).all() or (sd<=0).any():raise ValueError('Nonpositive cohort gene SD')
        scaled.loc[part.index,GENES]=(expression.to_numpy()-mu)/sd
        scalers[ds]={'mean':mu.tolist(),'sample_sd':sd.tolist(),'normalization_uses_labels':False}
    data['MMR4_LOCO']=np.nan;locks={}
    for target in sorted(data.dataset.unique()):
        training=scaled.loc[scaled.dataset.ne(target)].copy()
        fit=fit_cutoffs(training,grid)
        assert target not in fit['training_cohorts']
        mask=data.dataset.eq(target)
        data.loc[mask,'MMR4_LOCO']=expit((fit['cutoffs']-scaled.loc[mask,GENES].to_numpy())/.5).max(axis=1)
        fit['held_out_cohort']=target;fit['held_out_labels_used_for_fit']=False
        fit['held_out_scaler']=scalers[target];locks[target]=fit
    assert np.isfinite(data.MMR4_LOCO).all()
    return data,locks

def fuse_scores(subtype,mmr):
    subtype=np.asarray(subtype,dtype=float);mmr=np.asarray(mmr,dtype=float)
    if subtype.shape!=mmr.shape or not np.isfinite(subtype).all() or not np.isfinite(mmr).all():
        raise ValueError('Fusion scores must be aligned and finite')
    return .5*subtype+.5*mmr
