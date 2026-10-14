from pathlib import Path
import sys,json,copy,multiprocessing as mp,time
import numpy as np,pandas as pd
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P/'vendor'))
from pybayesprism.gibbs import GibbsSampler
from pybayesprism.joint_post import JointPost
from pybayesprism.theta_post import ThetaPost
from pybayesprism import optim
STREAM=None;OUTPUT=None

def sample_task(args):
 i,x,phi,alpha,indices,state,final,out=args
 path=Path(out)/f'{i:04d}.npz'
 if path.exists():
  z=np.load(path);return {k:z[k] for k in z.files}
 fun=GibbsSampler.sample_theta_n if final else GibbsSampler.sample_Z_theta_n
 r=fun(x,phi,alpha,indices,state)
 assert np.isfinite(r['theta_n']).all() and np.allclose(r['theta_n'].sum(),1,atol=1e-10)
 if not final:assert np.allclose(r['Z_n'].sum(axis=1),x,atol=1e-7)
 temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,**r);temp.replace(path)
 return r

def run_ref(self,final,compute_elbo):
 global STREAM,OUTPUT
 phi=self.reference.phi;xs=self.X.to_numpy();idx=GibbsSampler.get_gibbs_idx(self.gibbs_control)
 out=OUTPUT/('final_samples' if final else 'initial_samples');out.mkdir(exist_ok=True)
 tasks=[];states=[]
 for i in range(len(xs)):
  state=copy.deepcopy(STREAM.bit_generator.state);states.append(state)
  STREAM.bit_generator.advance(2**64)
  tasks.append((i,xs[i],phi,self.gibbs_control['alpha'],idx,state,final,str(out)))
 (out/'rng_states.json').write_text(json.dumps(states))
 (out/'identities.json').write_text(json.dumps({'sample_ids':self.X.index.tolist(),'genes':self.X.columns.tolist(),'types':phi.index.tolist()}))
 results=[];t=time.time()
 with mp.Pool(self.gibbs_control['n.cores']) as pool:
  for r in pool.imap(sample_task,tasks,chunksize=1):
   results.append(r);print(('final' if final else 'initial'),len(results),'/',len(xs),'elapsed',round(time.time()-t),flush=True)
 if final:return ThetaPost.new(self.X.index,phi.index,results)
 return JointPost.new(self.X.index,self.X.columns,phi.index,results)

def install(out):
 global STREAM,OUTPUT
 OUTPUT=Path(out);STREAM=np.random.default_rng(2026)
 GibbsSampler.run_gibbs_refPhi=run_ref
 GibbsSampler.estimate_gibbs_time=lambda *args,**kwargs: None
