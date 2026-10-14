# Bulk 7 versus 12 subtype models

The Excel workbook contains individual MSI-oriented prediction scores, development five-fold OOF scores, full-fit development apparent scores, and six-cohort retrospective evaluation. Cohort AUROC, balanced accuracy at 0.5, sensitivity, specificity and Brier score are provided for subtype-only, existing LOCO MMR4 and fixed equal-weight fusion.

This is a predeclared downstream comparison using historical BayesPrism subtype/parent inputs, the existing five folds and the current four-unit MLP settings. Selection is restricted to the 17 candidates with source scRNA oriented AUROC >0.70. The 7/12 panels require both development-oriented AUROCs >0.60/>0.50; source direction is frozen. Input order is the historical ten-feature order restricted to each panel, followed by additional candidates in ledger order. No retrospective outcomes select panels, hyperparameters or fusion weights.

Classifier scalers use training rows only, within each fold. Full-development feature screening and cohort-adaptive measurements remain outside CV. MMR4 uses the archived eight-cohort leave-one-cohort-out thresholds and per-cohort mean/sample-SD normalization, temperature 0.5. It is not end-to-end LOCO, and other retrospective cohorts enter each cutoff fit. These cohorts were previously viewed.

The 95% intervals use 2,000 paired MSI/MSS-stratified sample resamples of fixed scores. The RNG is seeded 2026 once and advances through the resampling; the actual indices are retained. These intervals do not include feature-selection, classifier-fitting or preprocessing uncertainty. GEO patient clustering has not been independently audited.

Source copies and SHA256 hashes, all 18 actual checkpoints (5 folds plus full fit for each 7/12 panel and a historical10 numerical control), independent verification, and plotted values are retained. No main manuscript or existing model is replaced.

Reproduce using Python 3.12.12, NumPy 1.26.4, pandas 2.3.3, SciPy 1.16.3 and scikit-learn 1.7.2:

    PYTHONHASHSEED=2026 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run_analysis.py --replay --output /tmp/bulk7_12_replay
    python verify_analysis.py /tmp/bulk7_12_replay

All English workbook tables are machine-readable; the main figure summarizes per-cohort subtype and fusion AUROCs.
