"""Publication-score regression oracle; predictions are not fitted in this test."""
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from reproduce_frozen_bulk import replay

HERE = Path(__file__).resolve().parent

class ArchivedScoreReplay(unittest.TestCase):
    def test_sample_scores_and_published_cohort_auroc(self):
        current, _ = replay(HERE / 'classifier_snapshot')
        saved = pd.read_csv(HERE / 'classifier_snapshot/predictions.csv', float_precision='round_trip')
        saved = saved[(saved.panel == 'panel12') & (saved.split != 'development_apparent')]
        saved = saved.set_index(['dataset','sampleID']).sort_index()
        actual = current.set_index(['dataset','sampleID']).sort_index()
        self.assertEqual(len(actual), 1212)
        self.assertTrue(actual.index.equals(saved.index))
        np.testing.assert_array_equal(actual.y, saved.y)
        np.testing.assert_allclose(actual[['Subtype','MMR4_LOCO','Fusion05']],saved[['Subtype','MMR4_LOCO','Fusion05']],rtol=0,atol=1e-12)
        metrics = pd.read_csv(HERE / 'classifier_snapshot/cohort_metrics.csv',float_precision='round_trip')
        metrics = metrics[(metrics.panel=='panel12') & (metrics.split!='development_apparent')]
        for cohort, part in actual.groupby(level=0):
            for component in ['Subtype','MMR4_LOCO','Fusion05']:
                archived = metrics[(metrics.dataset==cohort) & (metrics.component==component)].AUROC.item()
                self.assertAlmostEqual(roc_auc_score(part.y,part[component]),archived,places=12)

if __name__ == '__main__': unittest.main()
