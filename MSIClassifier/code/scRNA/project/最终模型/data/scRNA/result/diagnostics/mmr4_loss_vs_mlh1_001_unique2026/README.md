# Corrected unique-cell MMR4 and MLH1 comparison

Tumor and reference indices are combined by union, not concatenation. Each cell
enters CNV inference once. GSE178341 CNV was rerun with Stromal-only reference
and explicit PCA, neighbor and Leiden seeds of 2026. GSE132465 and GSE200997
already use the same corrected implementation. Historical outputs are retained.
The change from historical results includes explicit seed settings as well as
deduplication; the performance difference cannot be attributed solely to one.

## CNV and MLH1 audit

| Dataset | Unique analyzed cells | Stromal reference cells | q95 | Evaluated samples with candidates |
|---|---:|---:|---:|---:|
| GSE178341 | 111727 | 15436 | 0.0083527404 | 64/64 |
| GSE132465 | 50483 | 5932 | 0.0090740610 | 23/23 |
| GSE200997 | 31676 | 1303 | 0.0148275916 | 15/15 |

GSE178341 has 27710 candidate malignant epithelial cells in evaluated samples.
Tumor cells are sampled with a cap of 100000 before union with reference cells;
these are not all cells in the original h5ad. Samples require at least 200 total
cells; GSE200997 T_cac5 is excluded. There is no minimum malignant epithelial
cell count beyond having at least one cell for a measured sample mean.

Candidates must be tumor epithelial cells, not reference cells, with cluster-level
CNV score >= Stromal q95 minus 1e-12. This exploratory threshold is not an official
infercnvpy malignancy rule. Equality includes the boundary cluster and is a
substantive tie rule, not merely floating-point correction. The primary CNV output
also contains a historical q99 column; downstream MLH1 explicitly recomputes q95.
Measured zero expression is retained; no candidate cells would mean missing.

## Loss versus express target

The target is the local GSE178341 MLH1 label, with MSS assigned express: 22 loss
and 42 express, 64 samples from 62 patients. This four-gene classifier is not the
bulk MSI model or the original 18-subtype MSI model. Four-gene means use the same
candidate cells as the MLH1 baseline. MMR4 uses fold-specific StandardScaler and
L2 LogisticRegression (C=1, liblinear, unweighted, seed 2026), existing patient-grouped
five folds, and probability >=0.5. No tuning was performed. MLH1 predicts loss at
mean expression <0.01; its continuous AUROC uses negative expression.

| Scope | Method | AUROC | Accuracy | BA | TN | FP | FN | TP |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| All 64 | MLH1 <0.01 | 0.871212 | 0.796875 | 0.780303 | 35 | 7 | 6 | 16 |
| All 64 | MMR4 grouped OOF | 0.848485 | 0.781250 | 0.768398 | 34 | 8 | 6 | 16 |
| MSI only, 35 | MLH1 <0.01 | 0.736014 | 0.714286 | 0.709790 | 9 | 4 | 6 | 16 |
| MSI only, 35 | MMR4 grouped OOF | 0.723776 | 0.685714 | 0.671329 | 8 | 5 | 6 | 16 |

Historical all-sample AUROC values 0.967532 (MLH1) and 0.962121 (MMR4) are
superseded for this corrected analysis. MMR4 does not improve performance.

## External MSS-only screening flags

The final MMR4 loss classifier is fitted on all 64 development samples.

| Dataset | MSS samples | MMR4 flags | MLH1 <0.01 flags | Flagged samples |
|---|---:|---:|---:|---|
| GSE132465 | 19 | 1 | 1 | SMC05-T |
| GSE200997 | 13 | 2 | 2 | T_cac9, T_cac13 |

MMR4 loss probabilities are 0.658751, 0.624437 and 0.816665, respectively.
All three have measured mean MLH1=0, from 7, 29 and 10 candidate cells,
respectively. They must not be removed as missing values. These are false MSI
screening positives only if predicted loss is treated as MSI; the cohorts lack
confirmed MLH1 loss labels. They are not original 18-subtype model predictions.

## Interpretation and verification

Original clinical MMR_IHC metadata conflicts with local express labels for
C106T, C115T and C132T. Labels were not silently changed; see the copied,
unchanged original_annotation_audit.csv. The CNV analysis is cohort-wide, so the
grouped classifier OOF result is not fully nested validation of the whole pipeline.

Separately, replacing only MLH1 in the locked 18-subtype LDA MSI classifier gives
AUROC 0.974384 in GSE178341, 0.894737 in GSE132465 and 1.0 in GSE200997.
These MSI AUROCs have a different target from the loss/express table above;
the GSE178341 locked-model score is a development result, not new OOF performance.

Verification reproduced all five patient-grouped OOF folds, confusion matrices,
32 external MSS probabilities from the saved model, both threshold rules, and
sample counts. All three CNV branches passed cell uniqueness, reference,
candidate-count, expression-mean and AUROC checks. See verification.json and
the corresponding MLH1 impact directories. No manuscript was overwritten.

Updated editable vector PDF:
../mlh1_sample_expression_three_groups_unique2026/sample_MLH1_three_groups_editable.pdf
