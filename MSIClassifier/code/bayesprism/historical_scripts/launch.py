from pathlib import Path
import subprocess,os,json,time
from concurrent.futures import ThreadPoolExecutor,as_completed
import pandas as pd
P=Path(__file__).resolve().parent;O=P.parent
ledger=pd.read_csv(O/'input_audit.csv');datasets=ledger.sort_values('n').dataset.tolist() if 'n' in ledger else ledger.sort_values('n_samples').dataset.tolist()
env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONHASHSEED='2026',LC_ALL='en_US.utf8')
(O/'logs').mkdir(exist_ok=True)
def task(d):
 with open(O/'logs'/f'{d}.log','a') as f:
  p=subprocess.run(['/home/wyw/miniconda3/envs/MSI/bin/python','-u',str(P/'run_cohort.py'),d,'24'],stdout=f,stderr=subprocess.STDOUT,env=env)
 return d,p.returncode
if __name__=='__main__':
 result={};(O/'launch.json').write_text(json.dumps({'pid':os.getpid(),'cohorts':datasets,'parallel_cohorts':3,'cores_per_cohort':24,'start':time.time()},indent=2))
 with ThreadPoolExecutor(max_workers=3) as pool:
  for f in as_completed([pool.submit(task,d) for d in datasets]):
   d,rc=f.result();result[d]=rc;(O/'run_status.json').write_text(json.dumps(result,indent=2));print(d,rc,flush=True)
