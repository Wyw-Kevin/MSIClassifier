from pathlib import Path
import numpy as np,pandas as pd,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paths import EXPORT_ROOT as P, existing_export
P.mkdir(parents=True,exist_ok=True)
x=pd.read_csv(existing_export('major_reference_relative.csv'));sub=pd.read_csv(existing_export('all72_reference_relative.csv'))
cols=['Epi','Strom','TNKILC','Myeloid','B','Plasma','Mast']
assert len(x)==1270 and x.dataset.nunique()==9 and not x.duplicated(['dataset','sampleID']).any()
assert np.isfinite(x[cols]).all().all() and (x[cols]>=0).all().all() and np.allclose(x[cols].sum(axis=1),1)
order=['TCGA-COADREAD','GSE39582','GSE24550','GSE25071','GSE27544','GSE35896','GSE39084','GSE75316','GSE92921']
a=x.groupby('dataset')[cols].mean().loc[order]*100;a.insert(0,'n',x.groupby('dataset').size().reindex(order));a.to_csv(P/'major_cohort_mean_percent.csv')
d=[]
for (ds,g) in x.groupby('dataset'):
 for c in cols:
  v=g[c]*100;d.append(dict(dataset=ds,cell_type=c,mean_percent=v.mean(),median_percent=v.median(),q25_percent=v.quantile(.25),q75_percent=v.quantile(.75),min_percent=v.min(),max_percent=v.max(),SD_pp=v.std()))
pd.DataFrame(d).to_csv(P/'major_cohort_distribution_percent.csv',index=False)
scols=[c for c in sub if c not in ['dataset','sampleID']];sub.groupby('dataset')[scols].mean().loc[order].mul(100).to_csv(P/'subtype_cohort_mean_percent.csv')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained');colors=['#cf6a68','#d6b45f','#547eb5','#74ae8d','#a282bc','#e0995e','#8f959e'];bottom=np.zeros(9)
for c,color in zip(cols,colors):axes[0].bar(np.arange(9),a[c],bottom=bottom,label=c,color=color,width=.75);bottom+=a[c].to_numpy()
axes[0].set(xticks=np.arange(9),xticklabels=order,ylabel='Mean estimated relative fraction (%)',ylim=(0,100));axes[0].tick_params(axis='x',rotation=55);axes[0].legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,1.16),frameon=False,fontsize=9)
axes[1].boxplot([x.loc[x.dataset==ds,'Epi']*100 for ds in order],tick_labels=order,showfliers=True,patch_artist=True,boxprops={'facecolor':'#cf6a68','alpha':.65},medianprops={'color':'black'})
axes[1].set(ylabel='Estimated epithelial fraction (%)',ylim=(0,100));axes[1].tick_params(axis='x',rotation=55)
for ax,label in zip(axes,['A','B']):ax.text(-.1,1.04,label,transform=ax.transAxes,fontweight='bold',fontsize=15);ax.spines[['top','right']].set_visible(False)
for ext in ['png','pdf','svg']:fig.savefig(P/f'BayesPrism_cohort_composition.{ext}',dpi=180)
(P/'composition_verification.json').write_text(json.dumps({'samples':1270,'cohorts':9,'all_finite_nonnegative':True,'max_sum_error':float(np.abs(x[cols].sum(axis=1)-1).max()),'summaries':'arithmetic mean over patients, not pooled reads','excluded':'GSE18088'},indent=2))
print(a.round(2).to_string())
