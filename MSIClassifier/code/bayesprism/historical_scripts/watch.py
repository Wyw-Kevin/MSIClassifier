from pathlib import Path
import json,time,subprocess
P=Path(__file__).resolve().parent;O=P.parent
seen=None
while True:
 cohorts=[]
 for c in sorted((O/'cohorts').glob('*')):
  if not c.is_dir():continue
  cohorts.append({'dataset':c.name,'initial_samples_saved':len(list((c/'initial_samples').glob('*.npz'))),'final_samples_saved':len(list((c/'final_samples').glob('*.npz'))),'complete':(c/'complete.json').exists()})
 state={'expected_cohorts':9,'expected_samples':1270,'excluded_cohort':'GSE18088','seed':2026,'cohorts':cohorts,'updated_unix':time.time()}
 statuses=json.loads((O/'run_status.json').read_text()) if (O/'run_status.json').exists() else {}
 state['exit_codes']=statuses;state['status']='complete' if len(statuses)==9 and all(x==0 for x in statuses.values()) else ('finished_with_failures' if len(statuses)==9 else 'running')
 (O/'progress.json').write_text(json.dumps(state,indent=2))
 signature=[x['dataset'] for x in cohorts if x['complete']]
 if signature!=seen:
  subprocess.run(['/home/wyw/miniconda3/envs/MSI/bin/python',str(P/'postprocess.py')],check=True);seen=signature
 if len(statuses)==9:break
 time.sleep(30)
