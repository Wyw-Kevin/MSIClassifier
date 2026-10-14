from pathlib import Path
import pandas as pd,numpy as np,joblib,json
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
F=Path('/home/wyw/rstudio/MSI/最终模型');P=F/'report/article/pybayesprism9_seed2026'
f=[s for s in joblib.load(F/'model/reference/shared18_weighted_mlh1_2026/model.joblib')['features'] if s.startswith('c')];assert len(f)==18
sc=pd.read_csv(F/'report/article/bulk_decode70_direction_audit2026/singlecell_reference_direction.csv').set_index('feature').loc[f]
x=pd.read_csv(P/'subtype_over_major.csv');lab=pd.read_csv(F/'report/article/bulk_fusion_selection_20260930/input_features.csv')[['dataset','sampleID','MMR','y']]
x=x.merge(lab,on=['dataset','sampleID'],how='left',validate='one_to_one');assert len(x)==1270 and x.y.notna().all() and np.isfinite(x[f]).all().all()
assert (x.y==x.MMR.map({'MSI':1,'MSS':0})).all()
rows=[];errors=[]
for ds,g in x.groupby('dataset'):
 y=g.y.to_numpy().astype(int);n1=y.sum();n0=len(y)-n1
 for a in f:
  v=g[a].to_numpy();raw=roc_auc_score(y,v);check=(rankdata(v)[y==1].sum()-n1*(n1+1)/2)/(n1*n0);errors.append(abs(raw-check))
  direction=int(sc.loc[a,'sc_direction']);assert direction!=0
  auc=raw if direction==1 else 1-raw
  rows.append(dict(dataset=ds,feature=a,n=len(y),n_MSI=int(n1),sc_auc_msi_high=sc.loc[a,'sc_auc_msi_high'],sc_direction=direction,bulk_auc_msi_high=raw,bulk_auc_sc_direction=auc,same_direction=bool(auc>.5)))
t=pd.DataFrame(rows);assert max(errors)<1e-12
t.to_csv(P/'shared18_direction_AUROC_long.csv',index=False)
order=['TCGA-COADREAD','GSE39582','GSE24550','GSE25071','GSE27544','GSE35896','GSE39084','GSE75316','GSE92921']
piv=t.pivot(index='feature',columns='dataset',values='bulk_auc_sc_direction').loc[f,order];piv.to_csv(P/'shared18_direction_AUROC_matrix.csv')
s=t.groupby('dataset').agg(n=('n','first'),n_MSI=('n_MSI','first'),same_direction_count=('same_direction','sum')).loc[order]
s['features']=[','.join(piv.index[piv[d]>.5]) for d in order];s.to_csv(P/'shared18_direction_counts.csv')
res={}
for name,ds in [('development2',order[:2]),('external7',order[2:]),('all9',order)]:
 a=piv[ds];res[name]={'intersection_count':int((a>.5).all(axis=1).sum()),'intersection_features':a.index[(a>.5).all(axis=1)].tolist(),'macro_auc_gt050_count':int((a.mean(axis=1)>.5).sum())}
res.update({'reference':'GSE178341 64 patients; fixed direction from subtype/major fractions','n_samples':1270,'n_comparisons':len(t),'max_rank_AUC_error':max(errors),'rule':'strict AUROC >0.5 in scRNA direction; not significance; no external-direction flips; descriptive only; classifier unchanged'})
(P/'shared18_direction_verification.json').write_text(json.dumps(res,ensure_ascii=False,indent=2))
print(s.to_string());print(json.dumps(res,ensure_ascii=False,indent=2))
