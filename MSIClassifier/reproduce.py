"""CPU-only replay of archived features/models and conditional analyses.

This runner does not download data or train preprocessing/annotation models.
"""
from pathlib import Path
import argparse,json,subprocess,sys,time

ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branch',choices=['all','bulk','scRNA','spatial','paired','stability','bayesprism','figures'],default='all')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'output')
    args=parser.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    steps={
      'bulk':['code/bulk/reproduce_frozen_bulk.py','--output-dir',str(out/'bulk_replay')],
      'scRNA':['code/scRNA/replay_frozen_scRNA.py','--output-dir',str(out/'scRNA_replay')],
      'spatial':['code/spatial/reproduce_frozen_spatial.py','--output-dir',str(out/'spatial_replay')],
      'paired':['code/bulk/compute_paired_increment.py','--replay','--output',str(out/'paired_increment')],
      'stability':['code/bulk/compute_selection_stability.py','--output',str(out/'bulk_selection_stability')],
      'bayesprism':['code/bayesprism/validate_portable.py'],
      'figures':['code/reporting/reproduce_publication_figures.py','--output-dir',str(out/'figures')],
    }
    results=[]
    for branch in (steps if args.branch=='all' else [args.branch]):
        start=time.monotonic();log=out/(branch+'_run.log')
        print('Running',branch,flush=True)
        with log.open('w') as handle:
            completed=subprocess.run([sys.executable,*steps[branch]],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
        results.append({'branch':branch,'returncode':completed.returncode,'elapsed_seconds':round(time.monotonic()-start,3),'log':str(log.relative_to(out))})
        if completed.returncode:
            print(log.read_text()[-5000:]);raise SystemExit(completed.returncode)
        print('PASS:',branch,flush=True)
    (out/'replay_summary.json').write_text(json.dumps({'status':'passed','mode':'archived inputs; no refit','results':results},indent=2)+'\n')
    print('All requested replays passed. Output:',out)

if __name__=='__main__':main()
