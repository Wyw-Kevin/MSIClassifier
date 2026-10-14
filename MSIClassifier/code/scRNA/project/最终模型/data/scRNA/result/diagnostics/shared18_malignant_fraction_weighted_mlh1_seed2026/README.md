# Shared18 plus malignant-fraction weighted MLH1

Formula: weighted_MLH1 = raw malignant-epithelial mean MLH1 *
N_malignant_epithelial / N_all_epithelial. Counts use unique tumor cells retained
in the corrected cohort-specific Stromal-only q95 CNV analysis. This equals
summed candidate malignant-epithelial MLH1 divided by all epithelial cells; it
does not measure the actual expression in nonmalignant epithelial cells.
Actual zeros are retained; zero malignant cells would be missing. No clipping,
pseudocount, threshold search, extra feature selection or model tuning is used.

The historical 18 subtypes and classifier family are fixed: training-fitted
StandardScaler + LDA (lsqr, shrinkage=auto). Both raw and weighted controls are
refitted in GSE178341. Existing patient-grouped five folds are reused, with
scaler and classifier fitted inside each training fold. All configured seeds
are 2026. External cohorts are predicted with the final full-development fits.

| Dataset / evaluation | Raw refit AUROC | Weighted refit AUROC |
|---|---:|---:|
| GSE178341 apparent, 64 samples | 0.981281 | 0.989163 |
| GSE178341 grouped five-fold OOF | 0.954680 | 0.978325 |
| GSE132465 external, 23 samples | 0.881579 | 0.960526 |
| GSE200997 external, 15 samples | 1.000000 | 1.000000 |

At probability >=0.5, weighted GSE132465 has TN=17, FP=2, FN=0, TP=4;
accuracy=0.913043, BA=0.947368. The remaining false positives are SMC09-T
(0.989988) and SMC22-T (0.961961). Among the five previously highlighted MSS
samples, SMC01-T, SMC02-T and SMC17-T now have probabilities below 0.5.

These newly fitted models differ from the unchanged historical fitted model
whose replaced-MLH1 GSE132465 AUROC was 0.894737. Raw-refit predictions were
verified to exactly reproduce the previous inverse-weight experiment's control.

Limitations: GSE178341 counts are within sampled CNV inputs, not the full original
population. CNV was performed cohort-wide, features were historically selected,
and evaluation cohorts have been repeatedly examined. Trying formulas after
viewing external results is exploratory method development; these results do
not establish untouched independent validation or fully nested pipeline CV.
The weighting encodes both expression and estimated malignant epithelial
composition, rather than correcting expression measurement reliability.

Files: sample_counts_and_weights.csv contains all denominators and applied weights;
*_features.csv contains exact classifier inputs; sample_predictions.csv contains
all scores; model_metrics.csv includes AUROC, accuracy, BA and confusion matrices;
GSE132465_previous_five_false_positives.csv compares the five samples; separate
joblib files preserve both fitted models. Manifest records source and script hash.
Verification passed formula, bounded weights, zero retention, patient-disjoint
folds, OOF reproduction, saved-model prediction, AUROC and confusion-matrix checks.
Prior artifacts and manuscripts were not overwritten.
