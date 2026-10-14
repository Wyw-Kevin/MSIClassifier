"""Unlabelled inference must reproduce the actual full-fit probabilities."""
from pathlib import Path
import unittest,json
import numpy as np,pandas as pd,joblib
from predict_bulk import score

HERE=Path(__file__).resolve().parent

class DirectScoring(unittest.TestCase):
    def setUp(self):
        self.model=joblib.load(HERE/'classifier_snapshot/models/panel12_fold0.joblib')
        self.data=pd.read_csv(HERE/'classifier_snapshot/inputs/all_cohort_features.csv',float_precision='round_trip')

    def test_unlabelled_full_fit_matches_saved_scores(self):
        actual=score(self.data.drop(columns=['y','MMR']),self.model)
        saved=pd.read_csv(HERE/'classifier_snapshot/predictions.csv',float_precision='round_trip')
        saved=saved[(saved.panel=='panel12') & saved.split.isin(['development_apparent','retrospective'])]
        actual=actual.set_index(['dataset','sampleID']).sort_index();saved=saved.set_index(['dataset','sampleID']).sort_index()
        self.assertTrue(actual.index.equals(saved.index))
        np.testing.assert_allclose(actual.Subtype,saved.Subtype,rtol=0,atol=1e-12)

    def test_matching_lock_matches_saved_fusion(self):
        data=self.data[self.data.dataset.eq('GSE24550')].drop(columns=['y','MMR'])
        lock=json.loads((HERE/'mmr4/locks/heldout_GSE24550.json').read_text())
        actual=score(data,self.model,lock)
        saved=pd.read_csv(HERE/'classifier_snapshot/predictions.csv',float_precision='round_trip')
        saved=saved[saved.panel.eq('panel12') & saved.dataset.eq('GSE24550')].set_index('sampleID').loc[actual.sampleID]
        np.testing.assert_allclose(actual[['Subtype','MMR4_LOCO','Fusion05']],saved[['Subtype','MMR4_LOCO','Fusion05']],rtol=0,atol=1e-12)

    def test_missing_feature_and_unmatched_lock_stop_scoring(self):
        data=self.data[self.data.dataset.eq('GSE24550')].copy()
        with self.assertRaises(ValueError):score(data.drop(columns=[self.model['features'][0]]),self.model)
        lock=json.loads((HERE/'mmr4/locks/heldout_GSE24550.json').read_text());data['dataset']='new_cohort'
        with self.assertRaises(ValueError):score(data,self.model,lock)

if __name__=='__main__':unittest.main()
