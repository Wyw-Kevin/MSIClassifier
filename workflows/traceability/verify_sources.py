"""Verify complete label coverage and public-ID expression witnesses from saved data."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=HERE/'verification.json');args=p.parse_args()
    source=pd.read_csv(HERE/'data/S5_sample_label_sources.csv');included=source[source.included_primary]
    assert len(source)==1329 and len(included)==1328 and not source.duplicated(['dataset','analysis_sampleID']).any()
    assert source.analysis_label.eq(np.where(source.y.eq(1),'MSI','MSS')).all()
    bulk=pd.read_csv(REPO/'workflows/bulk/classifier_snapshot/predictions.csv')
    bulk=bulk[bulk.panel.eq('panel12')&bulk.split.isin(['development_OOF','retrospective'])].drop_duplicates(['dataset','sampleID'])
    sc=pd.read_csv(REPO/'workflows/scRNA/primary17/tables/all_predictions.csv')
    sc=sc[sc.model.eq('Joint')&sc.split.ne('training_apparent')]
    for preds,modality in [(bulk,'bulk'),(sc,'scRNA')]:
        target=included[included.modality.eq(modality)]
        matched=preds.merge(target,left_on=['dataset','sampleID'],right_on=['dataset','analysis_sampleID'],validate='one_to_one',suffixes=('_prediction','_source'))
        assert len(matched)==len(preds)==len(target) and matched.y_prediction.eq(matched.y_source).all()
    spatial=pd.read_csv(REPO/'workflows/scRNA/primary17/spatial/section_predictions.csv')
    target=included[included.modality.eq('spatial')].set_index('analysis_sampleID').loc[spatial.sampleID]
    assert len(target)==14 and np.array_equal(target.y,spatial.y)
    exceptions=source[source.exception_note.fillna('').str.contains('MSI-L')]
    assert len(exceptions)==10 and exceptions.dataset.eq('GSE24550').all() and exceptions.y.eq(0).all()
    excluded=source[~source.included_primary];assert excluded.analysis_sampleID.tolist()==['T_cac5']
    mapping=pd.read_csv(HERE/'data/S5_renamed_GEO_ID_mapping.csv')
    assert len(mapping)==81 and not mapping.duplicated(['dataset','analysis_sampleID']).any() and not mapping.public_GSM.duplicated().any()
    assert mapping.matching_gene_values.gt(.75*mapping.common_genes_compared).all()
    assert mapping.matching_gene_values.gt(10*mapping.next_best_matching_values).all()
    witness=pd.read_csv(HERE/'data/GEO_ID_expression_fingerprints.csv.gz')
    np.testing.assert_allclose(witness.raw_probe_mean,witness.stored_gene_expression,atol=1e-7,rtol=0)
    assert witness.groupby(['dataset','analysis_sampleID']).size().eq(128).all()
    joined=witness.merge(mapping[['dataset','analysis_sampleID','public_GSM']],on=['dataset','analysis_sampleID'],validate='many_to_one',suffixes=('','_map'))
    assert joined.public_GSM.eq(joined.public_GSM_map).all()
    proof={'passed':True,'analyzed_samples_or_sections':1328,'QC_exclusion':1,'renamed_GEO_samples':81,'expression_witness_values':len(witness),'primary_labels_match_saved_predictions':True,
        'archived_MSI_L_negative_records':10,'spatial_labels':'Public grouping inference; assay not confirmed','source_assay_gaps_not_inferred':True}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(proof,indent=2));print(json.dumps(proof,indent=2))
if __name__=='__main__':main()
