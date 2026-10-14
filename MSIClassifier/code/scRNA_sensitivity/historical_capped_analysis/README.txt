Completed exploratory scRNA sensitivity analysis, seed2026. Main classifier and its primary q95/0.65 identity preserved.
Subtype cutoffs0.60/0.65/0.70/0.75 use fixed historical full-development ranking, strict cutoff and cap18; q95 weighted MLH1 unchanged.
CNV q90/q95/q99 analyses keep all18 subtypes. Refit comparisons fit training-only scalers/LDA in archived patient folds; frozen-q95 comparisons change only measurement inputs.
External cohorts are scored without using labels to choose panels, quantiles or preprocessing. Historically viewed cohorts remain retrospective.
42 new fitted artifacts and1660 saved predictions passed independent saved-model, pairwise-AUROC and scaler/patient audits. Main-rule predictions reproduce archived scores within1e-10.
The original S2/S3 measurement tables and all previous supplemental tables are preserved. New S2/S3 performance extensions and S10 threshold sheets are in the English workbook.
Run run_analysis.py, then verify_analysis.py to reproduce numeric results; source_archive contains the required small inputs and original primary classifier. Delivery build uses the prior manuscript/supplement in the project.
