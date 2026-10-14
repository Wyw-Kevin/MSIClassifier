"""Scientific CPU replay; optional numeric figures, without refitting."""
from pathlib import Path
import argparse,json,subprocess,sys,os

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--branch',choices=['all','bulk','scRNA','sensitivity','spatial','paired','stability','bayesprism','plots'],default='all')
    p.add_argument('--output-dir',type=Path,default=ROOT/'output')
    p.add_argument('--plots',action='store_true')
    args=p.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    sense='workflows/scRNA/resources/article/scRNA_AUROC_threshold_all_candidates_seed2026_20261013/verify_analysis.py'
    steps={
      'bulk':['workflows/bulk/reproduce_frozen_bulk.py','--output-dir',str(out/'bulk')],
      'scRNA':['workflows/scRNA/replay_frozen_scRNA.py','--output-dir',str(out/'scRNA')],
      'sensitivity':[sense,'--output',str(out/'sensitivity/verification.json')],
      'spatial':['workflows/spatial/reproduce_frozen_spatial.py','--output-dir',str(out/'spatial')]+(['--figures'] if args.plots else []),
      'paired':['workflows/bulk/compute_paired_increment.py','--replay','--output',str(out/'paired')],
      'stability':['workflows/bulk/compute_selection_stability.py','--output',str(out/'stability')],
      'bayesprism':['workflows/bayesprism/validate_implementation.py','--output-dir',str(out/'bayesprism')],
    }
    if args.plots or args.branch=='plots':steps['plots']=['workflows/plots/reproduce_plots.py','--output-dir',str(out/'figures')]
    result=[];env=os.environ.copy();env.update(PYTHONHASHSEED='2026',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    for branch in (steps if args.branch=='all' else [args.branch]):
        print('Running',branch,flush=True)
        log=out/(branch+'.log')
        with log.open('w') as h:proc=subprocess.run([sys.executable,*steps[branch]],cwd=ROOT,env=env,stdout=h,stderr=subprocess.STDOUT)
        result.append({'branch':branch,'returncode':proc.returncode})
        if proc.returncode:print(log.read_text()[-5000:]);raise SystemExit(proc.returncode)
        print('PASS:',branch,flush=True)
    (out/'summary.json').write_text(json.dumps({'status':'passed','mode':'archived measurements/models, no refit','branches':result},indent=2)+'\n')

if __name__=='__main__':main()
