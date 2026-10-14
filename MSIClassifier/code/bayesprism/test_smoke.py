from pathlib import Path
import sys
P=Path(__file__).parent;sys.path.insert(0,str(P/'vendor'))
import numpy as np,pandas as pd
from pybayesprism import prism
import runtime
from paths import WORK_ROOT
out=WORK_ROOT/'synthetic_smoke';out.mkdir(parents=True,exist_ok=True);runtime.install(out)
genes=[f'g{i}' for i in range(1200)]
ref=np.ones((2,1200));ref[0,:600]=100;ref[1,600:]=100
ref=pd.DataFrame(ref,index=['A','B'],columns=genes)
x=pd.DataFrame(np.rint(np.array([[.2,.8],[.8,.2]])@ref.to_numpy()*10).astype(int),index=['one','two'],columns=genes)
p=prism.Prism.new(ref,'GEP',['A','B'],['A','B'],None,x)
r=p.run(n_cores=2,gibbs_control={'seed':2026,'chain.length':100,'burn.in':50,'thinning':2})
a=r.posterior_theta_f.theta
assert a.columns.tolist()==['A','B']
assert np.allclose(a.sum(axis=1),1)
assert np.max(abs(a.to_numpy()-[[.2,.8],[.8,.2]]))<.03
print(a);print('END-TO-END SMOKE PASS')
