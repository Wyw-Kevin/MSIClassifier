"""Validate copied provenance/resources and the active seed2026 import path; no real cohorts run."""
from pathlib import Path
import sys,json,hashlib,gzip,tempfile,subprocess,os,ast,platform,importlib.metadata
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P));sys.path.insert(0,str(P/'vendor'))
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    manifest=json.loads((P/'provenance_bayesprism.json').read_text())
    for record in manifest['source_records']:
        target=P/record['destination']
        assert digest(target)==record['packaged_sha256'],str(target)
        if record.get('decompressed_sha256'):
            with gzip.open(target,'rb') as source:
                assert hashlib.sha256(source.read()).hexdigest()==record['decompressed_sha256']
    import paths,runtime
    from pybayesprism import prism,process_input,optim,extract
    import pybayesprism.gibbs as gibbs
    assert Path(gibbs.__file__).resolve().is_relative_to((P/'vendor').resolve())
    assert prism.Prism.valid_gibbs_control({})['seed']==2026
    assert paths.selected_cohorts()==paths.CURRENT_COHORTS or os.environ.get('MSI_BAYESPRISM_COHORTS')
    for source in P.glob('*.py'):
        ast.parse(source.read_text(),filename=str(source))
    import test_port
    for fn in [test_port.test_one_type_conservation,test_port.test_symbol_categories,test_port.test_protein_filter_unknown_gene]:
        fn()
    import pandas as pd,numpy as np
    mapping=pd.read_csv(P/'resources/reference/mapping.csv',index_col=0)
    reference=pd.read_pickle(P/'resources/reference/raw_count_profiles.pkl')
    assert reference.shape==(72,17039) and np.isfinite(reference).all().all() and (reference>=0).all().all()
    assert (reference==np.rint(reference)).all().all()
    archived=pd.read_csv(P/'resources/historical_exports/subtype_over_major.csv').set_index(['dataset','sampleID'])
    errors=[]
    for cohort in paths.HISTORICAL_COHORTS:
        final=pd.read_csv(P/'resources/historical/cohorts'/cohort/'final_fractions.csv',index_col=0)
        assert final.shape[1]==72 and np.allclose(final.sum(axis=1),1)
        major=pd.DataFrame({parent:final[mapping.index[mapping.parent==parent]].sum(axis=1) for parent in mapping.parent.unique()})
        within=pd.DataFrame({key:final[key]/major[mapping.loc[key,'parent']] for key in final if key!='Epi'})
        errors.append(float((within-archived.loc[cohort].loc[within.index,within.columns]).abs().max().max()))
    assert max(errors)<1e-12
    (P/'verification').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='synthetic_',dir=P/'verification') as temporary:
        environment=os.environ.copy();environment['MSI_BAYESPRISM_WORKDIR']=temporary;environment['PYTHONHASHSEED']='2026'
        proc=subprocess.run([sys.executable,str(P/'test_smoke.py')],env=environment,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (P/'verification/synthetic_smoke.log').write_text(proc.stdout)
        assert proc.returncode==0 and 'END-TO-END SMOKE PASS' in proc.stdout,proc.stdout
    result={'status':'passed','copied_source_resources_verified':len(manifest['source_records']),
            'vendor_import_path':str(gibbs.__file__),'active_default_seed':2026,
            'reference_shape':list(reference.shape),'n_historical_cohorts':len(errors),
            'max_reconstructed_export_difference':max(errors),'one_type_conservation_and_gene_filter_checks':'passed',
            'two_type_synthetic_smoke':'passed','real_cohorts_computed':False,
            'verification_python_version':platform.python_version(),
            'verification_dependency_versions':manifest['verification_environment_versions'],
            'historical_runtime_versions_known':False}
    (P/'verification/portable_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
