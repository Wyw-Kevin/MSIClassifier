"""Test the actual patched port and bundled count reference; no real cohorts."""
from pathlib import Path
import argparse,tempfile,subprocess,sys,os,json
import numpy as np,pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'vendor'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',required=True,type=Path);args=p.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    from pybayesprism import prism
    import test_port
    assert prism.Prism.valid_gibbs_control({})['seed']==2026
    for f in [test_port.test_one_type_conservation,test_port.test_symbol_categories,test_port.test_protein_filter_unknown_gene]:f()
    ref=pd.read_pickle(HERE/'resources/reference/raw_count_profiles.pkl')
    assert ref.shape==(72,17039) and np.isfinite(ref).all().all() and (ref>=0).all().all()
    assert (ref==np.rint(ref)).all().all()
    with tempfile.TemporaryDirectory(prefix='synthetic_',dir=out) as temporary:
        env=os.environ.copy();env['MSI_BAYESPRISM_WORKDIR']=temporary;env['PYTHONHASHSEED']='2026'
        proc=subprocess.run([sys.executable,str(HERE/'test_smoke.py')],env=env,capture_output=True,text=True)
        (out/'synthetic.log').write_text(proc.stdout+proc.stderr)
        assert proc.returncode==0 and 'END-TO-END SMOKE PASS' in proc.stdout,proc.stdout+proc.stderr
    report={'status':'passed','active_seed':2026,'reference_shape':list(ref.shape),'one_type_and_filters':'passed','two_type_synthetic_mixture':'passed','real_cohorts_run':False}
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
