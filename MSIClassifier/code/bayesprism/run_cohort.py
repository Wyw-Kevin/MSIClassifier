from pathlib import Path
import sys,json,time,hashlib,pickle,random,os,fcntl
import numpy as np,pandas as pd
from paths import PACKAGE_ROOT as P, WORK_ROOT as O, reference_input, selected_cohorts
sys.path.insert(0,str(P/'vendor'))
import runtime
from pybayesprism import prism,extract

def run(d,cores):
 assert d in selected_cohorts(), 'Cohort is not enabled by MSI_BAYESPRISM_COHORTS'
 out=O/'cohorts'/d;out.mkdir(parents=True,exist_ok=True)
 lock=open(out/'run.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (out/'complete.json').exists():return
 random.seed(2026);np.random.seed(2026);runtime.install(out)
 ref=pd.read_pickle(reference_input('raw_count_profiles.pkl'));mat=pd.read_pickle(O/'inputs'/f'{d}_sampling.pkl');t=time.time()
 p=prism.Prism.new(ref,'GEP',ref.index.tolist(),ref.index.tolist(),None,mat)
 p.phi_cellType.phi.to_csv(out/'initial_reference.csv');p.mixture.to_pickle(out/'aligned_mixture.pkl')
 (out/'config.json').write_text(json.dumps({'dataset':d,'seed':2026,'key':None,'types':ref.index.tolist(),'n_samples':len(mat),'n_genes':p.mixture.shape[1],'chain':1000,'burn':500,'thin':2,'alpha':1,'sigma':2,'cores':cores,'port':'pybayesprism0.1.0 with documented audited fixes','linear_input_sha256':hashlib.sha256((O/'inputs'/f'{d}_linear.pkl').read_bytes()).hexdigest()},indent=2))
 r=p.run(n_cores=cores,update_gibbs=True,gibbs_control={'seed':2026,'chain.length':1000,'burn.in':500,'thinning':2,'alpha':1})
 for name,frame in [('initial',r.posterior_initial_cellState.theta),('initial_CV',r.posterior_initial_cellState.theta_cv),('final',r.posterior_theta_f.theta),('final_CV',r.posterior_theta_f.theta_cv)]:
  assert frame.shape==(len(mat),len(ref))
  assert np.isfinite(frame).all().all() and (frame>=0).all().all()
  if 'CV' not in name:assert np.allclose(frame.sum(axis=1),1,atol=1e-8)
  frame.index.name='sampleID';frame.to_csv(out/f'{name}_fractions.csv')
 r.reference_update.phi.to_csv(out/'updated_reference.csv')
 with open(out/'result.pkl','wb') as f:pickle.dump(r,f,protocol=5)
 (out/'complete.json').write_text(json.dumps({'dataset':d,'n':len(mat),'n_genes':p.mixture.shape[1],'n_types':len(ref),'elapsed_seconds':time.time()-t,'fractions_verified':True,'biological_validity_established':False},indent=2))
 print('COMPLETE',d,flush=True)
if __name__=='__main__':run(sys.argv[1],int(sys.argv[2]) if len(sys.argv)>2 else 24)
