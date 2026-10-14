"""Replay current spatial features and frozen classifier from bundled saved inputs.

No reference fitting, target fitting, posterior sampling or classifier fitting occurs.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import numpy as np
import pandas as pd
import joblib

HERE=Path(__file__).resolve().parent
SNAPSHOT=HERE/'resources'
BASE=SNAPSHOT/'transfer'
REPORT=SNAPSHOT/'report'
MODEL=HERE.parent/'scRNA/resources/model/reference/shared18_weighted_mlh1_2026/model.joblib'

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=HERE/'outputs/frozen_replay')
    parser.add_argument('--figures',action='store_true',help='Recreate the paired spatial plot from its retained source and saved inputs.')
    args=parser.parse_args(argv)
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    plan=json.loads((BASE/'analysis_plan.json').read_text())
    model_hash=hashlib.sha256(MODEL.read_bytes()).hexdigest()
    assert model_hash==plan['classifier_sha256'],'Current frozen classifier identity mismatch'
    artifact=joblib.load(MODEL)
    ecology=joblib.load(REPORT/'ecology_only_comparator.joblib')
    features=artifact['features'];pipeline=artifact['pipeline']
    assert len(features)==19 and features[-1]=='mean_MLH1_epithelial'
    assert pipeline['model'].solver=='lsqr' and pipeline['model'].shrinkage=='auto'
    groups=pd.read_csv(BASE/'reference/mapping.csv').set_index('small_type').major_type
    frozen_inputs=pd.read_csv(REPORT/'spatial_classifier_features.csv',index_col=0)
    expected_sections=pd.read_csv(REPORT/'section_predictions.csv').set_index('sampleID')
    expected_patients=pd.read_csv(REPORT/'patient_predictions.csv').set_index('patientID')
    audit=[]
    rebuilt_rows=[]
    for directory in sorted(BASE.glob('VISDS*')):
        sid=directory.name.split('(')[0]
        abundance=pd.read_csv(directory/'small_cell_abundance_means.csv',index_col=0)
        spots=pd.read_csv(directory/'spot_mlh1_and_abundance.csv.gz',index_col=0)
        assert abundance.index.equals(spots.index)
        posterior=np.load(directory/'posterior_means.npz')
        np.testing.assert_allclose(posterior['w_sf'],abundance.to_numpy(),atol=1e-7,rtol=1e-6)
        candidate=abundance['Epi_CNV_candidate'];other=abundance['Epi_other']
        reconstructed=np.log1p(10000*spots.candidate_MLH1_allocated_counts/spots.candidate_library_allocated_counts)
        np.testing.assert_allclose(reconstructed,spots.candidate_MLH1_reconstructed_log1p_cp10k,atol=1e-6,rtol=1e-5)
        weighted=float((reconstructed*candidate).sum()/(candidate+other).sum())
        totals=abundance.sum(axis=0)
        row={feature:float(totals[feature]/totals.loc[groups[groups.eq(groups.loc[feature])].index].sum()) for feature in features[:-1]}
        row['mean_MLH1_epithelial']=weighted
        row['sampleID']=sid;rebuilt_rows.append(row)
        np.testing.assert_allclose([row[f] for f in features],frozen_inputs.loc[sid,features].to_numpy(dtype=float),atol=1e-7,rtol=1e-5)
        history=pd.read_csv(directory/'history_elbo_train.csv',index_col=0).iloc[:,-1]
        assert len(history)==500 and np.isfinite(history).all()
        audit.append(dict(sampleID=sid,n_spots=len(spots),weighted_reconstructed=weighted,weighted_saved=float(frozen_inputs.loc[sid,'mean_MLH1_epithelial']),weighted_abs_error=abs(weighted-float(frozen_inputs.loc[sid,'mean_MLH1_epithelial'])),history_epochs=len(history)))
    rebuilt=pd.DataFrame(rebuilt_rows).set_index('sampleID')[features]
    assert len(rebuilt)==14
    # Prediction replay deliberately uses the exact archived feature table. The rebuilt
    # table above validates its meaning while retaining original float32/CSV provenance.
    X=frozen_inputs[features]
    section=pd.DataFrame({'sampleID':X.index,'score':pipeline.predict_proba(X)[:,1],
                          'ecology_only_score':ecology['pipeline'].predict_proba(X[ecology['features']])[:,1]})
    section['patientID']=section.sampleID.map(expected_sections.patientID)
    section['MMR']=section.sampleID.map(expected_sections.MMR)
    assert section.patientID.notna().all()
    reference=expected_sections.loc[section.sampleID]
    np.testing.assert_allclose(section.score,reference.score,atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(section.ecology_only_score,reference.ecology_only_score,atol=1e-12,rtol=1e-12)
    patient=section.groupby('patientID',as_index=False).agg(MMR=('MMR','first'),score=('score','mean'),ecology_only_score=('ecology_only_score','mean'),n_sections=('sampleID','size'))
    assert len(patient)==7 and patient.n_sections.eq(2).all()
    patient_reference=expected_patients.loc[patient.patientID]
    np.testing.assert_allclose(patient.score,patient_reference.score,atol=1e-12,rtol=1e-12)
    section.to_csv(output/'reproduced_section_predictions.csv',index=False)
    patient.to_csv(output/'reproduced_patient_predictions.csv',index=False)
    rebuilt.to_csv(output/'rebuilt_features_from_saved_spots.csv')
    pd.DataFrame(audit).to_csv(output/'saved_feature_reconstruction_audit.csv',index=False)
    figures=[]
    if args.figures:
        # Invoke only spatial(), never the legacy bulk()/publish()/main() operations.
        spec=importlib.util.spec_from_file_location('retained_spatial_plot_source',HERE/'plot_spatial.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.SOURCE=BASE;module.CURRENT=REPORT;module.OUT=output/'figures';module.OUT.mkdir(parents=True,exist_ok=True)
        vmax=module.spatial()
        display=pd.read_csv(module.OUT/'Figure04_spot_display_values.csv.gz',index_col=0)
        retained=pd.read_csv(SNAPSHOT/'Figure04_spot_display_values.csv.gz',index_col=0)
        assert len(display)==len(retained)==20733 and display.index.equals(retained.index)
        for column in ['candidate_abundance','other_epithelial_abundance','candidate_MLH1_reconstructed_log1p_cp10k','local_ecology_score','weighted_MLH1_display','epithelial_fraction']:
            np.testing.assert_allclose(display[column],retained[column],atol=1e-12,rtol=1e-12)
        np.testing.assert_array_equal(display.epithelial_enriched_display,retained.epithelial_enriched_display)
        figures=[str(p.relative_to(output)) for p in sorted(module.OUT.glob('*'))]
    proof=dict(passed=True,scope='Existing saved-input feature and frozen-weight replay; no fitting or posterior sampling',n_sections=len(section),n_patients=len(patient),n_spots=sum(r['n_spots'] for r in audit),classifier_sha256=model_hash,
               max_section_probability_error=float(np.max(np.abs(section.score.to_numpy()-reference.score.to_numpy()))),
               max_patient_probability_error=float(np.max(np.abs(patient.score.to_numpy()-patient_reference.score.to_numpy()))),
               max_weighted_feature_reconstruction_error=max(r['weighted_abs_error'] for r in audit),
               figure_display_values_verified=bool(args.figures),figure_outputs=figures,
               reference_epoch_history_available=False,convergence_established=False)
    (output/'verification.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2))
    print(json.dumps(proof,ensure_ascii=False,indent=2))
    return proof

if __name__=='__main__':main()
