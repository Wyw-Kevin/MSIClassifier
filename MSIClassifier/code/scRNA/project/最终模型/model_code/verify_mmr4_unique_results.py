"""Reproduce corrected grouped OOF and external MSS-only predictions."""

# Packaging-only path resolution; archived originals retain exact source bytes.
from pathlib import Path as _MSIPackagePath
import sys as _MSIPackageSys
for _MSIPackageParent in _MSIPackagePath(__file__).resolve().parents:
    if (_MSIPackageParent / "project_paths.py").is_file():
        _MSIPackageSys.path.insert(0, str(_MSIPackageParent))
        break
from project_paths import project_root
from pathlib import Path
import json
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score

BASE = Path(__file__).resolve().parents[1] / 'data/scRNA/result/diagnostics'
OUT = BASE / 'mmr4_loss_vs_mlh1_001_unique2026'
GENES = ['MLH1', 'MSH2', 'MSH6', 'PMS2']


def main():
    train = pd.read_csv(OUT / 'GSE178341_four_gene_features.csv', index_col=0)
    pred = pd.read_csv(OUT / 'training_grouped_OOF_predictions.csv', index_col=0)
    assert train.index.is_unique and pred.index.equals(train.index)
    assert train.groupby('patientID').fold.nunique().eq(1).all()
    for fold in train.fold.unique():
        fit = train[train.fold.ne(fold)]
        test = train[train.fold.eq(fold)]
        assert not set(fit.patientID) & set(test.patientID)
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=1, solver='liblinear', random_state=2026, max_iter=2000))
        model.fit(fit[GENES], fit.loss)
        np.testing.assert_allclose(model.predict_proba(test[GENES])[:, 1],
                                   pred.loc[test.index, 'MMR4_score'], atol=1e-7)
    np.testing.assert_array_equal(pred.MLH1_prediction, train.MLH1.lt(.01))
    np.testing.assert_array_equal(pred.MMR4_prediction, pred.MMR4_score.ge(.5))
    metrics = pd.read_csv(OUT / 'training_metrics.csv')
    for row in metrics.itertuples():
        part = pred if row.scope == 'all64' else pred[pred.MMR.eq('MSI')]
        np.testing.assert_allclose(roc_auc_score(part.loss, part[row.method + '_score']), row.AUROC)
        np.testing.assert_array_equal(
            confusion_matrix(part.loss, part[row.method + '_prediction'], labels=[0, 1]).ravel(),
            [row.TN, row.FP, row.FN, row.TP])
    ext = pd.read_csv(OUT / 'external_MSS_only_predictions.csv')
    assert not ext.duplicated(['dataset', 'sampleID']).any()
    assert ext.MMR.eq('MSS').all() and ext[GENES].notna().all().all()
    assert ext.n_malignant_epithelial.gt(0).all()
    saved = joblib.load(OUT / 'MMR4_loss_model.joblib')
    np.testing.assert_allclose(saved['pipeline'].predict_proba(ext[GENES])[:, 1],
                               ext.MMR4_loss_probability, atol=1e-7)
    np.testing.assert_array_equal(ext.MMR4_loss_flag, ext.MMR4_loss_probability.ge(.5))
    np.testing.assert_array_equal(ext.MLH1_001_loss_flag, ext.MLH1.lt(.01))
    assert ext.groupby('dataset').size().to_dict() == {'GSE132465': 19, 'GSE200997': 13}
    summary = ext.groupby('dataset').agg(
        MSS_samples=('sampleID', 'size'), MMR4_loss_flags=('MMR4_loss_flag', 'sum'),
        MLH1_001_loss_flags=('MLH1_001_loss_flag', 'sum'))
    pd.testing.assert_frame_equal(summary, pd.read_csv(OUT / 'external_MSS_false_flags.csv', index_col=0))
    # Original clinical annotation does not depend on the recomputed CNV scores.
    shutil.copy2(BASE / 'mmr4_loss_vs_mlh1_001/original_annotation_audit.csv',
                 OUT / 'original_annotation_audit.csv')
    report = dict(passed=True, grouped_folds=int(train.fold.nunique()),
                  training_samples=len(train), patients=train.patientID.nunique(),
                  external_MSS_samples=len(ext), score_tolerance=1e-7)
    (OUT / 'verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
