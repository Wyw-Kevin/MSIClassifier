import os
from pathlib import Path
from zipfile import ZipFile
from copy import deepcopy
from lxml import etree as E
import pandas as pd,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics import roc_curve,roc_auc_score
import json,re,hashlib,shutil
HERE=Path(__file__).resolve().parent
A=HERE.parent/'scRNA/resources/article';O=Path(os.environ.get('MSI_FIGURE_OUTPUT',str(HERE/'outputs')));O.mkdir(parents=True,exist_ok=True)
sources=[A/'restore_feature_selection_20260928/Figure02_source.csv',A/'weighted_scRNA_update/AUROC_with_patient_bootstrap_CI.csv',A/'weighted_scRNA_update/all_model_predictions.csv',HERE.parent/'scRNA/resources/diagnostics/shared18_malignant_fraction_weighted_mlh1_seed2026/sample_counts_and_weights.csv']
grid,metrics,preds,counts=[pd.read_csv(p) for p in sources]
order=['gradient_boosting','gradient_boosting_shallow','logistic_l2','logistic_l2_balanced','linear_svc','linear_svc_balanced','ridge_classifier','lda_shrinkage','mlp_4_l2','perceptron_balanced','sgd_log_l2']
labels=['Gradient boosting','Shallow boosting','Logistic regression','Balanced logistic','Linear SVM','Balanced linear SVM','Ridge classifier','Shrinkage LDA','MLP','Balanced perceptron','SGD logistic']
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'svg.fonttype':'none','pdf.fonttype':42})
fig=plt.figure(figsize=(6.8,9.0));gs=fig.add_gridspec(4,2,left=.105,right=.985,bottom=.07,top=.96,hspace=.64,wspace=.42,height_ratios=[1,1,1.12,1.12])
axes=[fig.add_subplot(gs[0,:]),fig.add_subplot(gs[1,:]),fig.add_subplot(gs[2,0]),fig.add_subplot(gs[2,1]),fig.add_subplot(gs[3,0]),fig.add_subplot(gs[3,1])]
for ax,(ds,title) in zip(axes[:2],[('GSE178341','A  Subtype selection: GSE178341 OOF'),('GSE132465','B  Subtype selection: GSE132465')]):
 pos=ax.get_position();ax.set_position([.255,pos.y0,.70,pos.height])
 q=grid[grid.dataset==ds];m=q.pivot(index='model',columns='n_subtype_features',values='auc').reindex(index=order,columns=range(10,25));assert m.shape==(11,15) and not m.isna().any().any()
 im=ax.imshow(m,cmap='YlGnBu',vmin=.5,vmax=1,aspect='auto')
 ax.set_xticks(range(15),range(10,25));ax.set_yticks(range(11),labels,fontsize=7);ax.tick_params(length=0,pad=2)
 ax.set_xlabel('Subtype count (+ whole-epithelial MLH1; historical)',fontsize=7.5,labelpad=3);ax.set_title(title,loc='left',fontweight='bold',pad=6)
 for i in range(11):
  for j in range(15):ax.text(j,i,f'{m.iloc[i,j]:.2f}',ha='center',va='center',fontsize=5.7,color='white' if m.iloc[i,j]>.85 else 'black')
 ax.add_patch(Rectangle((7.5,6.5),1,1,fill=False,edgecolor='#C8392B',linewidth=1.4))
 cax=fig.add_axes([.967,pos.y0,.009,pos.height]);bar=fig.colorbar(im,cax=cax,ticks=[.5,.75,1]);bar.ax.tick_params(labelsize=5.5,length=1,pad=1)
 for spine in ax.spines.values():spine.set_visible(False)
colors=['#277F8E','#B65066','#6C8A44'];scopes=[('GSE178341','training_apparent'),('GSE178341','grouped_OOF'),('GSE132465','external'),('GSE200997','external')]
ax=axes[2]
for off,model,color in [(-.12,'18 subtypes + raw MLH1','#888888'),(.12,'18 subtypes + weighted MLH1',colors[0])]:
 vals=[metrics[(metrics.dataset==ds)&(metrics.split==sp)&(metrics.model==model)].iloc[0] for ds,sp in scopes]
 y=np.array([v.AUROC for v in vals]);lo=np.array([v.CI_low for v in vals]);hi=np.array([v.CI_high for v in vals])
 ax.errorbar(np.arange(4)+off,y,yerr=[y-lo,hi-y],fmt='o',markersize=3,capsize=2,color=color,label='Unweighted MLH1' if off<0 else 'Weighted MLH1')
ax.set_xticks(range(4),['GSE178341\nApparent','GSE178341\nOOF','GSE132465\nRetrospective','GSE200997\nRetrospective'],fontsize=5.5)
ax.set(ylabel='AUROC',ylim=(.5,1.04));ax.legend(frameon=False,fontsize=6.5,loc='lower left');ax.set_title('C  MLH1 weighting comparison',loc='left',fontweight='bold')
ax=axes[3]
for (ds,sp),color in zip(scopes[1:],colors):
 q=preds[(preds.dataset==ds)&(preds.split==sp)&(preds.model=='18 subtypes + weighted MLH1')]
 x,y,_=roc_curve(q.y,q.score);ax.plot(x,y,color=color,label=f'{ds}: {roc_auc_score(q.y,q.score):.3f}',linewidth=1)
ax.plot([0,1],[0,1],ls='--',color='#aaaaaa',lw=.7);ax.set(xlabel='False positive rate',ylabel='True positive rate',xlim=(-.02,1.02),ylim=(-.02,1.02));ax.legend(frameon=False,fontsize=6,loc='lower right');ax.set_title('D  Weighted-model ROC',loc='left',fontweight='bold')
ax=axes[4];rng=np.random.default_rng(2026)
for i,ds in enumerate(['GSE178341','GSE132465','GSE200997']):
 for j,cl in enumerate(['MSS','MSI']):
  vals=counts[(counts.dataset==ds)&(counts.MMR==cl)].weighted_MLH1.to_numpy();pos=i+(j-.5)*.28;color=colors[j]
  ax.boxplot([vals],positions=[pos],widths=.22,showfliers=False,patch_artist=True,boxprops={'facecolor':color,'alpha':.3},medianprops={'color':'black'})
  ax.scatter(pos+rng.uniform(-.05,.05,len(vals)),vals,s=5,color=color,alpha=.7,label=cl if i==0 else None)
ax.set_xticks(range(3),['GSE178341','GSE132465','GSE200997'],fontsize=6.2);ax.set_ylabel('Fraction-weighted MLH1',fontsize=8);ax.legend(frameon=False,fontsize=6.5,loc='upper right');ax.set_title('E  Sample-level weighted MLH1',loc='left',fontweight='bold')
ax=axes[5];models=['18 subtypes + weighted MLH1','18 subtypes only','Weighted MLH1 only']
for i,(ds,sp) in enumerate(scopes[1:]):
 vals=[metrics[(metrics.dataset==ds)&(metrics.split==sp)&(metrics.model==m)].iloc[0] for m in models]
 y=np.array([v.AUROC for v in vals]);lo=np.array([v.CI_low for v in vals]);hi=np.array([v.CI_high for v in vals])
 ax.errorbar(np.arange(3)+(i-1)*.2,y,yerr=[y-lo,hi-y],fmt='o',markersize=3,color=colors[i],capsize=2,label=ds)
ax.set_xticks(range(3),['18 subtypes\n+ weighted MLH1','18 subtypes\nonly','Weighted\nMLH1 only'],fontsize=6.4);ax.set(ylabel='AUROC',ylim=(.3,1.04));ax.legend(frameon=False,fontsize=6.5,loc='lower left');ax.set_title('F  Component ablation',loc='left',fontweight='bold')
for ax in axes[2:]:ax.spines[['top','right']].set_visible(False)
for ext in ['png','svg','pdf']:fig.savefig(O/('Figure02_model_development_evaluation.'+ext),dpi=260)
plt.close(fig)

