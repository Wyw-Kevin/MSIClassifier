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
A=Path('/home/wyw/rstudio/MSI/最终模型/report/article');O=A/'merged_scrna_figure_20260928';O.mkdir(exist_ok=True)
f=A/'MMR多转录组分类器_第二版.docx';backup=O/'MMR多转录组分类器_第二版_合并图2图3前.docx';assert not backup.exists()
sources=[A/'restore_feature_selection_20260928/Figure02_source.csv',A/'weighted_scRNA_update/AUROC_with_patient_bootstrap_CI.csv',A/'weighted_scRNA_update/all_model_predictions.csv',A.parents[1]/'data/scRNA/result/diagnostics/shared18_malignant_fraction_weighted_mlh1_seed2026/sample_counts_and_weights.csv']
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
 ax.set_xlabel('Subtype count (+ one original MLH1 feature)',fontsize=7.5,labelpad=3);ax.set_title(title,loc='left',fontweight='bold',pad=6)
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

W='http://schemas.openxmlformats.org/wordprocessingml/2006/main';R='http://schemas.openxmlformats.org/officeDocument/2006/relationships';ns={'w':W}
with ZipFile(f) as z:infos=z.infolist();blobs={i.filename:z.read(i.filename) for i in infos}
r=E.fromstring(blobs['word/document.xml']);before=deepcopy(r);body=r.find('w:body',ns)
text=lambda p:''.join(p.xpath('.//w:t/text()',namespaces=ns))
def put(p,s):
 ts=p.xpath('.//w:t',namespaces=ns);ts[0].text=s
 for n in ts[1:]:n.text=''
selection_caption=None;changes=[]
for p in list(body.findall('w:p',ns)):
 old=text(p)
 if p.xpath('.//*[local-name()="docPr"][@descr="图2"]'):
  body.remove(p);continue
 if old.startswith('图2.'):
  selection_caption=deepcopy(p);body.remove(p);continue
 def remap(m):
  n=int(m[1]);letter=m[2] or '';tail=m[3] or ''
  if n==3:
   conv={'A':'C','B':'D','C':'E','D':'F'};return '图2'+conv.get(letter,letter)+(tail[0]+conv.get(tail[1],tail[1]) if tail else '')
  return '图'+str(n-1)+letter+tail
 new=re.sub(r'图([3-6])([A-D])?([–-][A-D])?(?!\d)',remap,old)
 # Existing figure 2 refers to the selection panels, now part of the combined figure.
 if old and not old.startswith('图3'):new=new.replace('（图2）','（图2A–B）')
 if old.startswith('图3A.'):
  put(selection_caption,'图2. 单细胞模型的构建与性能评价。（A–B）10–24个亚型与11类分类器在GSE178341五折OOF及GSE132465中的AUROC比较；各组合另含原定义的MLH1特征，红框标记保留的18亚型收缩LDA配置。A–B采用原MinMax及未占比加权的MLH1方案，沿用原始种子20260713的结果；C–F采用Z-score及当前MLH1定义，随机种子为2026。GSE132465曾参与开发阶段比较；前期亚型筛选及队列级CNV分析未在当前交叉验证各训练折内重复。')
  p.addprevious(selection_caption)
 # Subpanel legends continue under a single Figure 2 heading.
 if old.startswith(('图3A.','图3B.','图3C.','图3D.')):new=re.sub(r'^图2([C-F])\. ',lambda m:'（'+m[1]+'）',new)
 if new!=old:put(p,new);changes.append({'before':old,'after':new})
 for n in p.xpath('.//*[local-name()="docPr"]'):
  desc=n.get('descr','')
  if desc=='图3':
   n.set('descr','图2');n.set('name','Combined single-cell model figure')
   ex=p.xpath('.//*[local-name()="extent"]')[0];cx=int(ex.get('cx'));cy=round(cx*9/6.8)
   for v in p.xpath('.//*[local-name()="extent" or local-name()="ext"]'):
    if v.get('cx') is not None and v.get('cy') is not None:v.set('cx',str(cx));v.set('cy',str(cy))
   # Allow caption to flow to the following page rather than forcing both onto one page.
   pr=p.find('w:pPr',ns)
   if pr is not None:
    for tag in ['keepNext','keepLines']:
     el=pr.find('w:'+tag,ns)
     if el is not None:pr.remove(el)
  elif re.fullmatch(r'图[4-6]',desc):n.set('descr','图'+str(int(desc[1:])-1))
assert selection_caption is not None
for ext,part in [('png','word/media/image5.png'),('svg','word/media/image6.svg')]:blobs[part]=(O/('Figure02_model_development_evaluation.'+ext)).read_bytes()
rels=E.fromstring(blobs['word/_rels/document.xml.rels'])
for n in list(rels):
 if n.get('Id') in ['rIdSelectionPNG','rIdSelectionSVG']:rels.remove(n)
for ext in ['png','svg']:del blobs['word/media/subtype_selection.'+ext]
blobs['word/document.xml']=E.tostring(r,xml_declaration=True,encoding='UTF-8',standalone=True);blobs['word/_rels/document.xml.rels']=E.tostring(rels,xml_declaration=True,encoding='UTF-8',standalone=True)
assert [E.tostring(n) for n in r.xpath('//w:tbl',namespaces=ns)]==[E.tostring(n) for n in before.xpath('//w:tbl',namespaces=ns)]
assert [E.tostring(n) for n in r.xpath('//*[local-name()="oMath"]')]==[E.tostring(n) for n in before.xpath('//*[local-name()="oMath"]')]
assert len(r.xpath('//w:drawing',namespaces=ns))==5
assert '本图用于交代特征选择过程' not in text(r)
shutil.copy2(f,backup);tmp=f.with_suffix('.merge.tmp')
with ZipFile(tmp,'w') as z:
 for i in infos:
  if i.filename in blobs:z.writestr(i,blobs[i.filename])
with ZipFile(tmp) as z:
 assert z.testzip() is None
 for n in z.namelist():
  if n.endswith(('.xml','.rels','.svg')):E.fromstring(z.read(n))
tmp.replace(f)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report={'status':'passed','sha256':sha(f),'figure_count':5,'panel_mapping':{'old2A':'2A','old2B':'2B','old3A':'2C','old3B':'2D','old3C':'2E','old3D':'2F','old4':'3','old5':'4','old6':'5'},'tables_and_formulas_preserved':True,'retraining':False,'sources':{str(p):sha(p) for p in sources},'rendering':'Scientific figure rendered; full Word pagination not rendered; old PDF not updated'}
(O/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));(O/'changes.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2))
shutil.copy2(__file__,O/'merge_figures.py')
print(json.dumps(report,ensure_ascii=False,indent=2))
