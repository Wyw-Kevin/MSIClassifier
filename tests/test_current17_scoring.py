"""Public scoring contract for the current17 model and existing example measurements."""
from pathlib import Path
import json,subprocess,sys,tempfile,unittest,hashlib
import pandas as pd
import numpy as np
import joblib
ROOT=Path(__file__).resolve().parents[1]
SCORER=ROOT/'workflows/scRNA/predict_scrna.py'
MODEL=ROOT/'workflows/scRNA/resources/model/reference/auc070_17_weighted_mlh1_2026/model.joblib'
class Current17(unittest.TestCase):
    def score(self,frame):
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)/'input.csv';out=Path(tmp)/'scores.csv';frame.to_csv(src,index=False)
            proc=subprocess.run([sys.executable,str(SCORER),'--feature-table',str(src),'--output',str(out)],capture_output=True,text=True)
            return proc,pd.read_csv(out) if out.exists() else None
    def test_example_uses_current17_without_cM04_and_keeps_model_frozen(self):
        artifact=joblib.load(MODEL);self.assertEqual(len(artifact['features']),18);self.assertNotIn('cM04',artifact['features'])
        before=hashlib.sha256(MODEL.read_bytes()).hexdigest()
        frame=pd.read_csv(ROOT/'examples/scRNA_features.csv');self.assertNotIn('cM04',frame)
        proc,out=self.score(frame);self.assertEqual(proc.returncode,0,proc.stderr)
        expected=pd.read_csv(ROOT/'examples/scRNA_expected_scores.csv')
        np.testing.assert_allclose(out.score,expected.score,atol=1e-12,rtol=0)
        self.assertEqual(hashlib.sha256(MODEL.read_bytes()).hexdigest(),before)
    def test_missing_measured_subtype_stops(self):
        frame=pd.read_csv(ROOT/'examples/scRNA_features.csv').drop(columns='cTNI14')
        proc,out=self.score(frame);self.assertNotEqual(proc.returncode,0);self.assertIn('cTNI14',proc.stderr);self.assertIsNone(out)
    def test_outcome_metadata_does_not_change_scores(self):
        frame=pd.read_csv(ROOT/'examples/scRNA_features.csv');proc,out=self.score(frame)
        frame['MMR']='arbitrary metadata';proc2,out2=self.score(frame)
        self.assertEqual(proc.returncode,0,proc.stderr);self.assertEqual(proc2.returncode,0,proc2.stderr)
        np.testing.assert_array_equal(out.score,out2.score)
if __name__=='__main__':unittest.main()
