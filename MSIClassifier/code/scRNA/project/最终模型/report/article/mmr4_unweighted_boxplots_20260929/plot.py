from pathlib import Path as _PP
import sys as _PS
for _PD in _PP(__file__).resolve().parents:
 if (_PD/"project_paths.py").is_file():_PS.path.insert(0,str(_PD));break
from project_paths import project_root
from pathlib import Path
import pandas as pd,numpy as np,matplotlib,json,hashlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
B=(project_root()/'最终模型');O=B/'report/article/mmr4_unweighted_boxplots_20260929';S=B/'data/scRNA/result/diagnostics/mmr4_loss_vs_mlh1_001_unique2026'
genes=['MLH1','MSH2','MSH6','PMS2'];datasets=['GSE178341','GSE132465','GSE200997'];frames=[];sources={}
counts=pd.read_csv(B/'data/scRNA/result/diagnostics/shared18_malignant_fraction_weighted_mlh1_seed2026/sample_counts_and_weights.csv')
for ds,expected in zip(datasets,[64,23,15]):
 p=S/(ds+'_four_gene_features.csv');df=pd.read_csv(p);assert len(df)==expected and df.sampleID.is_unique
 assert df.MMR.isin(['MSI','MSS']).all() and df.n_malignant_epithelial.gt(0).all()
 assert np.isfinite(df[genes].to_numpy()).all() and df[genes].ge(0).all().all()
 c=counts[counts.dataset.eq(ds)].set_index('sampleID').loc[df.sampleID]
 assert np.array_equal(c.MMR.to_numpy(),df.MMR.to_numpy())
 np.testing.assert_allclose(c.raw_MLH1,df.MLH1,rtol=1e-7,atol=1e-10)
 np.testing.assert_array_equal(c.n_malignant,df.n_malignant_epithelial)
 df['dataset']=ds;frames.append(df[['dataset','sampleID','MMR','n_malignant_epithelial']+genes]);sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
wide=pd.concat(frames,ignore_index=True);long=wide.melt(id_vars=['dataset','sampleID','MMR','n_malignant_epithelial'],value_vars=genes,var_name='gene',value_name='mean_expression');long.to_csv(O/'plot_source.csv',index=False)
summary=long.groupby(['gene','dataset','MMR']).mean_expression.agg(n='size',median='median',minimum='min',maximum='max');summary.to_csv(O/'group_summary.csv')
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.titlesize':13,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,'svg.fonttype':'none','pdf.fonttype':42})
fig,axs=plt.subplots(2,2,figsize=(10.5,7.5));fig.subplots_adjust(left=.10,right=.985,bottom=.105,top=.88,hspace=.43,wspace=.29)
colors={'MSS':'#277F8E','MSI':'#B65066'};rng=np.random.default_rng(2026)
for letter,gene,ax in zip('ABCD',genes,axs.flat):
 for i,ds in enumerate(datasets):
  for j,label in enumerate(['MSS','MSI']):
   vals=wide.loc[wide.dataset.eq(ds)&wide.MMR.eq(label),gene].to_numpy();pos=i+(j-.5)*.34
   ax.boxplot([vals],positions=[pos],widths=.27,showfliers=False,whis=1.5,patch_artist=True,boxprops={'facecolor':colors[label],'edgecolor':colors[label],'alpha':.3},medianprops={'color':'#242424','linewidth':1.2},whiskerprops={'color':colors[label]},capprops={'color':colors[label]})
   ax.scatter(pos+rng.uniform(-.07,.07,len(vals)),vals,s=17,color=colors[label],alpha=.75,edgecolors='white',linewidths=.25,zorder=3)
 ax.set_xticks(range(3),datasets);ax.set_xlim(-.5,2.5);ax.set_ylabel('Mean normalized expression');ax.set_title(gene,fontstyle='italic',pad=12)
 ax.text(-.16,1.075,letter,transform=ax.transAxes,fontfamily='DejaVu Sans',fontsize=15,fontweight='bold',va='bottom',ha='left')
 ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15,linewidth=.5);ax.set_axisbelow(True)
 lo,hi=ax.get_ylim();ax.set_ylim(-.04*hi,hi*1.05)
fig.legend(handles=[Line2D([0],[0],marker='o',linestyle='',label=k,color=v,markersize=6) for k,v in colors.items()],loc='upper center',bbox_to_anchor=(.53,.98),ncol=2,frameon=False)
fig.text(.52,.025,'Samples (MSS / MSI):  GSE178341 29 / 35;  GSE132465 19 / 4;  GSE200997 13 / 2',ha='center',fontsize=9)
for ext in ['png','svg','pdf']:fig.savefig(O/('MMR4_unweighted_sample_expression.'+ext),dpi=260)
plt.close(fig)
(O/'图注.md').write_text('四个MMR基因在三个单细胞队列中的样本级表达分布。（A–D）依次为MLH1、MSH2、MSH6及PMS2。横坐标为GSE178341、GSE132465和GSE200997，每个队列内分别展示MSS与MSI。纵坐标为同一CNV保留细胞集合中候选恶性上皮细胞的归一化基因表达均值，不乘候选恶性上皮比例，不进行样本间Z-score，也不合成为MMR4评分。每个散点代表一个肿瘤样本；箱体表示第25至第75百分位，中线为中位数，须线延伸至1.5倍四分位距内的最远观测值，全部观测值以散点保留。三个队列分别纳入64、23和15例，其MSS/MSI例数分别为29/35、19/4和13/2；实测零表达保留。各基因采用独立纵轴范围，图用于描述分布，不据此判定增加基因的预测增量。\n')
(O/'verification.json').write_text(json.dumps({'samples':len(wide),'observations':len(long),'gene_panels':genes,'cohorts':datasets,'label_source':'MMR column (MSI/MSS), not loss/express','sample_counts_and_MLH1_match_current_model_inputs':True,'proportion_weighting':False,'between_sample_zscore':False,'random_seed_jitter':2026,'source_sha256':sources},ensure_ascii=False,indent=2))
print(summary.to_string())
