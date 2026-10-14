from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd,anndata as ad
from scipy import sparse
from paths import PACKAGE_ROOT as P, WORK_ROOT as O, PROJECT_ROOT as R, reference_input, selected_cohorts
F=R/'最终模型';O.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(P/'vendor'))
from pybayesprism import process_input
(O/'reference').mkdir(exist_ok=True);(O/'inputs').mkdir(exist_ok=True)
cm=pd.read_csv(reference_input('cell_manifest.csv.gz'),index_col=0)
assert cm.CellType.nunique()==72 and cm.groupby('CellType').size().min()>=50
x=ad.read_h5ad(R/'GSE178341.h5ad',backed='r');idx=x.obs_names.get_indexer(cm.index);assert (idx>=0).all()
order=np.argsort(idx);idx=idx[order];cm=cm.iloc[order]
genes=pd.Index(x.var_names.astype(str));unique=pd.Index(genes.unique());gidx=unique.get_indexer(genes)
gmap=sparse.csr_matrix((np.ones(len(genes)),(np.arange(len(genes)),gidx)),shape=(len(genes),len(unique)))
types=sorted(cm.CellType.unique());tidx=pd.Index(types).get_indexer(cm.CellType)
sums=np.zeros((len(types),len(unique)));expressed=np.zeros(len(unique),dtype=np.int64)
for j in range(0,len(idx),5000):
 # counts may be memory-backed by anndata even when X is backed; use same raw layer throughout.
 raw=x.layers['counts'][idx[j:j+5000],:];raw=sparse.csr_matrix(raw)@gmap
 assert np.isfinite(raw.data).all() and (raw.data>=0).all()
 expressed+=np.asarray((raw>0).sum(axis=0)).ravel()
 a=sparse.csr_matrix((np.ones(raw.shape[0]),(tidx[j:j+5000],np.arange(raw.shape[0]))),shape=(len(types),raw.shape[0]))
 sums+= (a@raw).toarray()
 if j%50000==0:print('reference cells',j,flush=True)
x.file.close()
ref=pd.DataFrame(sums,index=types,columns=unique);ref=ref.loc[:,expressed>=5]
ref=process_input.cleanup_genes(ref,'GEP','hs',['Rb','Mrp','other_Rb','chrM','MALAT1','chrX','chrY'],0)
ref=process_input.select_gene_type(ref,['protein_coding']);ref.to_pickle(O/'reference/raw_count_profiles.pkl')
mp=cm.groupby('CellType',sort=True).agg(n_cells=('CellType','size'),parent=('clTopLevel','first'))
mp.to_csv(O/'reference/mapping.csv');cm.to_csv(O/'reference/cells.csv')
pd.DataFrame({'gene':unique,'n_expressing_cells':expressed,'selected':unique.isin(ref.columns)}).to_csv(O/'reference/gene_audit.csv',index=False)
ledger=pd.read_csv(reference_input('sample_manifest.csv'));audit=[]
for d,g in ledger.groupby('dataset'):
 if d not in selected_cohorts():continue
 p=R/'bulk_data'/d/'expression.csv';mat=pd.read_csv(p,index_col=0).loc[:,g.sampleID].T
 assert mat.columns.is_unique and mat.index.is_unique
 bad=~np.isfinite(mat).all(axis=0);pd.Series(mat.columns[bad],name='excluded_nonfinite_gene').to_csv(O/'inputs'/f'{d}_excluded_genes.csv',index=False);mat=mat.loc[:,~bad]
 assert mat.min().min()>=0
 if d=='TCGA-COADREAD':mat=np.exp2(mat)-1;tr='2**x-1; linear RSEM normalized expression'
 elif d=='GSE92921':tr='unchanged MAS5 linear intensity'
 else:mat=np.exp2(mat);tr='2**x; linear intensity proxy from processed gene-level log2 expression'
 assert np.isfinite(mat).all().all() and mat.min().min()>=0
 mat.to_pickle(O/'inputs'/f'{d}_linear.pkl')
 # Multinomial sampling requires integer totals. Preserve continuous input; explicitly round once.
 rounded=np.rint(mat).astype(np.int64);rounded.to_pickle(O/'inputs'/f'{d}_sampling.pkl')
 audit.append(dict(dataset=d,n=len(mat),genes=mat.shape[1],reference_overlap=len(ref.columns.intersection(mat.columns)),transform=tr,rounding='numpy.rint for multinomial; intensity proxies are not read counts',max_rounding_error=float(np.abs(rounded-mat).max().max()),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
pd.DataFrame(audit).to_csv(O/'input_audit.csv',index=False)
(O/'prepare_complete.json').write_text(json.dumps({'n_cohorts':len(audit),'n_samples':sum(a['n'] for a in audit),'reference_cells':len(cm),'types':len(types),'reference_genes':ref.shape[1],'seed':2026,'key':None,'no_MSI_labels_loaded':True},indent=2))
print((O/'prepare_complete.json').read_text(),flush=True)
