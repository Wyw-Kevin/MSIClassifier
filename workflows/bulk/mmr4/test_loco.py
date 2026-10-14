import itertools
import unittest
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from loco import fit_cutoffs, loco_predictions, fuse_scores, GENES

class LocoTest(unittest.TestCase):
    def setUp(self):
        self.grid = np.array(list(itertools.product([-1., 0.], repeat=4)))
        rows = []
        for k, ds in enumerate(['a', 'b', 'c']):
            values = np.array([[2,1,2,1],[1,2,1,2],[0,-2,0,-1],[-2,0,-1,0]],float)
            values[:, k] += np.array([0,.2,-.4,.1])
            for j, v in enumerate(values):
                rows.append(dict(dataset=ds,sampleID=ds+str(j),y=int(j>=2),**dict(zip([g+'_raw' for g in GENES],v))))
        self.data = pd.DataFrame(rows)

    def test_heldout_labels_do_not_select_their_thresholds(self):
        a, locks = loco_predictions(self.data, self.grid)
        changed = self.data.copy()
        changed.loc[changed.dataset=='b','y'] = 1-changed.loc[changed.dataset=='b','y']
        b, altered = loco_predictions(changed, self.grid)
        np.testing.assert_array_equal(a.loc[a.dataset=='b','MMR4_LOCO'],b.loc[b.dataset=='b','MMR4_LOCO'])
        np.testing.assert_array_equal(locks['b']['cutoffs'],altered['b']['cutoffs'])
        self.assertEqual(locks['b']['training_cohorts'],['a','c'])

    def test_optimizer_matches_independent_scalar_search(self):
        training = self.data[self.data.dataset!='c'].copy()
        for ds, p in training.groupby('dataset'):
            raw=p[[g+'_raw' for g in GENES]]
            training.loc[p.index,GENES]=(raw-raw.mean()).to_numpy()/raw.std(ddof=1).to_numpy()
        candidates=[]
        for cut in self.grid:
            av=[]; bv=[]
            for ds,p in training.groupby('dataset'):
                score=expit((cut-p[GENES].to_numpy())/.5).max(axis=1)
                av.append(roc_auc_score(p.y,score));bv.append(balanced_accuracy_score(p.y,score>=.5))
            candidates.append((-round(np.mean(av),12),-round(np.mean(bv),12),*cut))
        expected=np.array(sorted(candidates)[0][2:])
        actual=fit_cutoffs(training,self.grid)
        np.testing.assert_array_equal(actual['cutoffs'],expected)

    def test_fusion_is_equal_weight_without_mutation(self):
        subtype=np.array([.1,.6,.8]);mmr=np.array([.9,.2,.4]);copy=subtype.copy()
        np.testing.assert_allclose(fuse_scores(subtype,mmr),[.5,.4,.6])
        np.testing.assert_array_equal(subtype,copy)

if __name__=='__main__':unittest.main()
