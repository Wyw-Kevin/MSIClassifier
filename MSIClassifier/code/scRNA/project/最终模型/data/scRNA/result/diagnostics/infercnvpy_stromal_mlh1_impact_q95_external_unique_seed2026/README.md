# External stromal-only q95 MLH1 evaluation

The frozen TOSICA major annotation supplies the Stromal reference independently
within each cohort. Normal epithelial cells are not reference cells. Reference
and tumor cell lists are combined by unique cell ID; overlapping cells are not
duplicated. PCA, neighbors and Leiden use random_state=2026.

CNV inference retains the existing expression scale and window100/step10
settings. The q95 threshold is calculated from reference-cell cluster-level CNV
scores. Candidate malignant epithelial cells are tumor epithelial cells outside
the reference set whose score is >= q95 - 1e-12. This is a study-defined rule,
not an official infercnvpy malignancy cutoff. Including ties is intentional;
it is not merely a correction of CSV precision.

GSE132465 epithelial identity uses the original broad Epithelial cells annotation;
GSE200997 uses frozen TOSICA Epi. MLH1 is the original h5ad X expression mean
among candidate cells. No minimum candidate count beyond one is imposed; zero
means missing rather than zero expression. The GSE200997 174-cell T_cac5 sample
is excluded by the existing >=200 total-cell QC rule.

The historical shared18_LDA_zscore2026 classifier, its scaler and the original
18 subtype feature values are unchanged. No classifier retraining or external
label-based threshold selection was performed. GSE200997's original MLH1 was
an epithelial proxy; it was not confirmed malignant-epithelial expression.

| Cohort | Stromal reference | q95 | Scored samples | Original AUROC, same samples | New AUROC |
|---|---:|---:|---:|---:|---:|
| GSE132465 | 5932 | 0.009074061 | 23/23 (4 MSI) | 0.921053 | 0.894737 |
| GSE200997 | 1303 | 0.014827592 | 15/15 (2 MSI) | 1.000000 | 1.000000 |

No eligible sample lacks candidate malignant epithelial cells. Candidate counts
in evaluated samples total 13648 and 5337, respectively.

Historical reference experiments concatenated overlapping tumor/reference lists
without deduplication and did not explicitly set PCA/neighbor seeds. Consequently,
the current results include implementation corrections and are not a pure
reference-only contrast with those experiments. Historical GSE178341 AUROC
0.9852 has not been rerun under these corrections and must not be presented as
an identically processed training result alongside this external evaluation.

Files: per-cohort sample_MLH1 tables contain counts and old/new expression;
selected_epithelial_cells.csv.gz files preserve individual candidate IDs;
sample_predictions.csv and locked_model_metrics.csv preserve scores and AUROCs;
verification.json records independently recomputed thresholds, counts, means and
AUROCs. Source CNV outputs are in ../infercnvpy_stromal_only_unique_seed2026.
The source CNV file also contains a historical q99 flag; that flag is NOT used
in this q95 evaluation. The manuscript and locked model remain unchanged.
