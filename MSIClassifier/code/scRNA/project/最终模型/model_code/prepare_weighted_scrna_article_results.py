"""Matched ablations, patient bootstrap and editable figure for weighted scRNA."""

# Packaging-only path resolution; archived originals retain exact source bytes.
from pathlib import Path as _MSIPackagePath
import sys as _MSIPackageSys
for _MSIPackageParent in _MSIPackagePath(__file__).resolve().parents:
    if (_MSIPackageParent / "project_paths.py").is_file():
        _MSIPackageSys.path.insert(0, str(_MSIPackageParent))
        break
from project_paths import project_root
from pathlib import Path
import json
import subprocess
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

FINAL = project_root() / '最终模型'
BASE = FINAL / 'data/scRNA/result/diagnostics'
SOURCE = BASE / 'shared18_malignant_fraction_weighted_mlh1_seed2026'
OUT = FINAL / 'report/article/weighted_scRNA_update'
OUT.mkdir(exist_ok=True)
VARIANT = 'malignant_fraction_weighted_MLH1_refit'
artifact = joblib.load(SOURCE / f'{VARIANT}.joblib')
features = artifact['features']
ledger = pd.read_csv(BASE / 'shared18_same_sample_benchmark_seed2026/sample_ledger.csv')
ledger = ledger[ledger.dataset.eq('GSE178341')].set_index('sampleID')
counts = pd.read_csv(SOURCE / 'sample_counts_and_weights.csv')
tables = {ds: pd.read_csv(SOURCE / f'{ds}_{VARIANT}_features.csv', index_col=0)
          for ds in counts.dataset.unique()}
train = tables['GSE178341']
y = counts[counts.dataset.eq('GSE178341')].set_index('sampleID').MMR.eq('MSI').astype(int).loc[train.index]
basepred = pd.read_csv(SOURCE / 'sample_predictions.csv')
preds = [basepred.assign(model=basepred.variant.map({VARIANT: '18 subtypes + weighted MLH1',
                                                 'raw_MLH1_refit': '18 subtypes + raw MLH1'}))]
for name, cols in [('18 subtypes only', features[:-1]), ('Weighted MLH1 only', features[-1:])]:
    rows = []
    for fold in sorted(ledger.fold.unique()):
        test = ledger.loc[train.index].fold.eq(fold)
        model = clone(artifact['pipeline']).fit(train.loc[~test, cols], y.loc[~test])
        for sid, p in zip(train.index[test], model.predict_proba(train.loc[test, cols])[:, 1]):
            rows.append(dict(model=name, dataset='GSE178341', split='grouped_OOF', sampleID=sid, y=int(y.loc[sid]), score=p))
    model = clone(artifact['pipeline']).fit(train[cols], y)
    for ds, table in tables.items():
        labels = counts[counts.dataset.eq(ds)].set_index('sampleID').MMR.eq('MSI').astype(int)
        for sid, p in zip(table.index, model.predict_proba(table[cols])[:, 1]):
            rows.append(dict(model=name, dataset=ds, split='training_apparent' if ds=='GSE178341' else 'external',
                             sampleID=sid, y=int(labels.loc[sid]), score=p))
    preds.append(pd.DataFrame(rows))
preds = pd.concat(preds, ignore_index=True)
preds.to_csv(OUT / 'all_model_predictions.csv', index=False)
rng = np.random.default_rng(2026)
metrics, deltas = [], []
for (ds, split), part in preds.groupby(['dataset','split']):
    wide = part.pivot(index='sampleID', columns='model', values='score')
    target = part.drop_duplicates('sampleID').set_index('sampleID').y.loc[wide.index]
    patient = ledger.loc[wide.index, 'patientID'] if ds=='GSE178341' else pd.Series(wide.index, index=wide.index)
    frame = pd.DataFrame({'patient':patient, 'y':target})
    assert frame.groupby('patient').y.nunique().eq(1).all()
    groups = [frame.loc[frame.y.eq(c),'patient'].unique() for c in [0,1]]
    indexmap = {pid: np.flatnonzero(patient.to_numpy()==pid) for pid in patient.unique()}
    boot = []
    for _ in range(2000):
        sampled = np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
        ids = np.concatenate([indexmap[pid] for pid in sampled])
        boot.append([roc_auc_score(target.iloc[ids],wide.iloc[ids][model]) for model in wide.columns])
    boot = pd.DataFrame(boot,columns=wide.columns)
    for model in wide.columns:
        lo,hi = boot[model].quantile([.025,.975])
        metrics.append(dict(dataset=ds, split=split, model=model, n=len(wide),
            AUROC=roc_auc_score(target,wide[model]), CI_low=lo, CI_high=hi))
    for baseline in ['18 subtypes only','Weighted MLH1 only','18 subtypes + raw MLH1']:
        diff = boot['18 subtypes + weighted MLH1']-boot[baseline]
        deltas.append(dict(dataset=ds,split=split,baseline=baseline,
            delta=roc_auc_score(target,wide['18 subtypes + weighted MLH1'])-roc_auc_score(target,wide[baseline]),
            CI_low=diff.quantile(.025),CI_high=diff.quantile(.975)))
metrics=pd.DataFrame(metrics); metrics.to_csv(OUT/'AUROC_with_patient_bootstrap_CI.csv',index=False)
pd.DataFrame(deltas).to_csv(OUT/'paired_ablation_AUROC_differences.csv',index=False)
font_manager.fontManager.addfont(subprocess.check_output(['fc-match','-f','%{file}','Arial'],text=True).strip())
plt.rcParams.update({'font.family':'Arial','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,axes=plt.subplots(2,2,figsize=(10,10));fig.subplots_adjust(left=.10,right=.97,bottom=.09,top=.95,wspace=.37,hspace=.45)
colors=['#277F8E','#B65066','#6C8A44']
ax=axes[0,0]
scopes=[('GSE178341','training_apparent'),('GSE178341','grouped_OOF'),('GSE132465','external'),('GSE200997','external')]
for offset,model,color in [(-.12,'18 subtypes + raw MLH1','#888888'),(.12,'18 subtypes + weighted MLH1',colors[0])]:
    vals=[metrics[(metrics.dataset==d)&(metrics.split==s)&(metrics.model==model)].iloc[0] for d,s in scopes]
    x=np.arange(4)+offset; z=np.array([v.AUROC for v in vals]);low=np.array([v.CI_low for v in vals]);high=np.array([v.CI_high for v in vals])
    ax.errorbar(x,z,yerr=[z-low,high-z],fmt='o',color=color,capsize=3,label=model.replace('18 subtypes + ',''))
ax.set_xticks(range(4),['GSE178341\nApparent','GSE178341\nGrouped OOF','GSE132465\nRetrospective','GSE200997\nRetrospective'],fontsize=8)
ax.set(ylabel='AUROC',ylim=(.5,1.04)); ax.legend(frameon=False,fontsize=8,loc='lower left');ax.set_title('A  Raw versus weighted MLH1',loc='left',fontweight='bold')
ax=axes[0,1]
for (ds,split),color in zip([('GSE178341','grouped_OOF'),('GSE132465','external'),('GSE200997','external')],colors):
    q=preds[(preds.dataset==ds)&(preds.split==split)&(preds.model=='18 subtypes + weighted MLH1')]
    fpr,tpr,_=roc_curve(q.y,q.score);ax.plot(fpr,tpr,color=color,label=f'{ds}: {roc_auc_score(q.y,q.score):.3f}')
ax.plot([0,1],[0,1],ls='--',color='#aaaaaa',lw=.8);ax.set(xlabel='False positive rate',ylabel='True positive rate',xlim=(-.02,1.02),ylim=(-.02,1.02));ax.legend(frameon=False,fontsize=8,loc='lower right');ax.set_title('B  Weighted-model ROC curves',loc='left',fontweight='bold')
ax=axes[1,0]
for i,ds in enumerate(['GSE178341','GSE132465','GSE200997']):
    for j,cl in enumerate(['MSS','MSI']):
        vals=counts[(counts.dataset==ds)&(counts.MMR==cl)].weighted_MLH1.to_numpy();pos=i+(j-.5)*.28;color=['#277F8E','#B65066'][j]
        ax.boxplot([vals],positions=[pos],widths=.22,showfliers=False,patch_artist=True,boxprops={'facecolor':color,'alpha':.3},medianprops={'color':'black'})
        ax.scatter(pos+rng.uniform(-.05,.05,len(vals)),vals,s=10,color=color,alpha=.7,label=cl if i==0 else None)
ax.set_xticks(range(3),['GSE178341','GSE132465','GSE200997'],fontsize=8);ax.set_ylabel('Malignant-fraction weighted MLH1');ax.legend(frameon=False,fontsize=8);ax.set_title('C  Sample-level weighted MLH1',loc='left',fontweight='bold')
ax=axes[1,1]
models=['18 subtypes + weighted MLH1','18 subtypes only','Weighted MLH1 only']
for i,(ds,split) in enumerate([('GSE178341','grouped_OOF'),('GSE132465','external'),('GSE200997','external')]):
    vals=[metrics[(metrics.dataset==ds)&(metrics.split==split)&(metrics.model==model)].iloc[0] for model in models]
    x=np.arange(3)+(i-1)*.2;z=np.array([v.AUROC for v in vals]);lo=np.array([v.CI_low for v in vals]);hi=np.array([v.CI_high for v in vals])
    ax.errorbar(x,z,yerr=[z-lo,hi-z],fmt='o',color=colors[i],capsize=3,label=ds)
ax.set_xticks(range(3),['18 subtypes\n+ weighted MLH1','18 subtypes\nonly','Weighted\nMLH1 only'],fontsize=8);ax.set(ylabel='AUROC',ylim=(.3,1.04));ax.legend(frameon=False,fontsize=8,loc='lower left');ax.set_title('D  Matched-model ablations',loc='left',fontweight='bold')
for ax in axes.flat:ax.spines[['top','right']].set_visible(False)
for ext in ['png','pdf','svg']:fig.savefig(OUT/f'Figure03_weighted_scRNA.{ext}',dpi=220)
plt.close(fig)
print(metrics.to_string(index=False))
print(pd.DataFrame(deltas).query("dataset == 'GSE178341' and split == 'grouped_OOF'").to_string(index=False))
