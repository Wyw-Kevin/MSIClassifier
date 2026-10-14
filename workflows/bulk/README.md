# Bulk

`predict_bulk.py` accepts unlabelled feature tables and applies the actual full-development panel12 StandardScaler/4-unit MLP. It optionally applies an explicit matching cohort's MMR4 cutoff/scaler lock and fixed0.5/0.5 fusion. It rejects an unmatched lock. No universal new-cohort MMR4 lock was established by the eight-round evaluation.

`reproduce_frozen_bulk.py` replays1212 stored measurements with six actual panel12 checkpoints: development folds1–5 and full-fit retrospective fold0. It validates all saved scores and recomputes cohort/macro AUROC. `compute_paired_increment.py --replay --output output/paired` recomputes true fusion-minus-MMR4 paired intervals. `compute_selection_stability.py --output output/stability` independently reproduces the exact2000 development draws and17-candidate selection frequencies. `mmr4/loco.py` is the actual training-only cutoff search, with held-out-label isolation/tie tests.

Run tests:

```sh
python workflows/bulk/test_direct_scoring.py
python workflows/bulk/test_archived_score_replay.py
python workflows/bulk/mmr4/test_loco.py
```

Refit the stored7/12 comparison (not raw deconvolution, not default replay):

```sh
python workflows/bulk/classifier_snapshot/run_analysis.py --replay --output output/bulk_refit
python workflows/bulk/classifier_snapshot/verify_analysis.py output/bulk_refit
```

The comparison retains a historical10 numerical control required by its original integrity check. It is not the default model. Stored source copies used by that run are necessary input snapshots, not executable historical pipelines. The selected12 features are cS27,cTNI16,cM06,cTNI05,cTNI10,cP1,cTNI13,cS18,cTNI04,cS17,cM05,cTNI08. Positive outcome coding is1. Current seeds2026, existing five folds and fixed fusion weights are preserved.

To refit all eight MMR4 rounds from stored log inputs, import the executable optimizer:

```python
from itertools import product
import sys, numpy as np, pandas as pd
sys.path.insert(0, "workflows/bulk/mmr4")
from loco import loco_predictions
frame = pd.read_csv("workflows/bulk/classifier_snapshot/inputs/all_cohort_features.csv", float_precision="round_trip")
grid = np.array(list(product(np.arange(-4.0, 0.01, 0.5), repeat=4)))
predictions, locks = loco_predictions(frame, grid)
```

The optimizer uses labels from the seven complement cohorts for each round, ranks macro-AUROC and BA after12-decimal rounding, and chooses the lexicographically smallest tied cutoff vector. This is explicit threshold fitting, separate from default frozen-lock replay.
