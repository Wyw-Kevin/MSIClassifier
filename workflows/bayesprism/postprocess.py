from pathlib import Path
import json,hashlib
import pandas as pd,numpy as np
from paths import PACKAGE_ROOT as P, WORK_ROOT as O, EXPORT_ROOT as E, reference_input
E.mkdir(parents=True,exist_ok=True)
mp=pd.read_csv(reference_input('mapping.csv'),index_col=0);whole=[];parents=[];within=[];summary=[];verification=[]
for d in pd.read_csv(O/'input_audit.csv').dataset:
 c=O/'cohorts'/d
 if not (c/'complete.json').exists():continue
 x=pd.read_csv(c/'final_fractions.csv',index_col=0);assert set(x)==set(mp.index)
 ids=pd.read_pickle(O/'inputs'/f'{d}_sampling.pkl').index;assert x.index.tolist()==ids.tolist()
 assert np.isfinite(x).all().all() and (x>=0).all().all() and np.allclose(x.sum(axis=1),1)
 major=pd.DataFrame({p:x[mp.index[mp.parent==p]].sum(axis=1) for p in mp.parent.unique()})
 sub=pd.DataFrame({f:x[f]/major[mp.loc[f,'parent']].replace(0,np.nan) for f in x if f!='Epi'})
 assert np.array_equal(major.Epi.to_numpy(),x.Epi.to_numpy())
 for frame,coll in [(x,whole),(major,parents),(sub,within)]:
  frame=frame.copy();frame.insert(0,'dataset',d);frame.index.name='sampleID';coll.append(frame.reset_index())
 summary.append(dict(dataset=d,n=len(x),Epi_mean_percent=x.Epi.mean()*100,Epi_min_percent=x.Epi.min()*100,Epi_max_percent=x.Epi.max()*100,Epi_SD_pp=x.Epi.std()*100,median_subtype_CV_percent=(sub.std()/sub.mean()).median()*100))
 verification.append(json.loads((c/'complete.json').read_text()))
for coll,name in [(whole,'all72_reference_relative.csv'),(parents,'major_reference_relative.csv'),(within,'subtype_over_major.csv')]:
 if coll:pd.concat(coll,ignore_index=True).to_csv(E/name,index=False)
pd.DataFrame(summary).to_csv(E/'cohort_summary.csv',index=False)
(O/'input_audit.csv').replace(E/'input_audit.csv') if False else None
pd.read_csv(O/'input_audit.csv').to_csv(E/'input_audit.csv',index=False)
(E/'verification.json').write_text(json.dumps({'completed_cohorts':len(verification),'completed_samples':sum(x['n'] for x in verification),'expected_cohorts':len(pd.read_csv(O/'input_audit.csv')),'expected_samples':int(pd.read_csv(O/'input_audit.csv')['n'].sum()),'all_complete':len(verification)==len(pd.read_csv(O/'input_audit.csv')),'details':verification},indent=2))
(E/'说明.md').write_text('''# BayesPrism outputs

The final posterior-mean fractions are in final_fractions.csv. Parent-relative subtype features are in subtype_over_major.csv. Fixed raw-count reference, independent target-cohort updates, chain settings and seed2026 follow the workflow README. The continuous target input and the integer sampling input are both retained in the run directory. Microarray linear intensities remain platform-specific proxies.

MSI labels are not used to adapt the reference. These outputs are conditional computational estimates; their generation does not establish measurement equivalence.
''')
print('Postprocess',len(verification),'cohorts',flush=True)
