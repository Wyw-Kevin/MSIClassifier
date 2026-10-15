# Troubleshooting and result interpretation

Start from bulk gene/probe expression or scRNA/spatial expression files: [Expression input workflow](EXPRESSION_WORKFLOW.md). The existing tables below describe stored-feature scoring and archived replay where stated.

[Main README](../README.md) · [Input format](INPUT_FILES.md)

## Installation and file paths

| Symptom | Cause/action |
| --- | --- |
| `ModuleNotFoundError` | Activate the intended Python 3.12 environment and install its `requirements.txt` |
| Model file cannot be found | Keep model/resources with the supplied directory layout, or explicitly pass a compatible `--model` |
| Input file cannot be found | Commands assume the repository working directory; use correct relative/absolute paths and quote paths with spaces |
| Incompatible scikit-learn/joblib deserialization | Use the pinned tested versions and the supplied/trusted artifact; do not substitute a different serialized format |
| `fc-match` missing during paired spatial plots | Install fontconfig, or replay scores without `--figures`/spatial `--plots` |
| BayesPrism missing h 5ad/expression | Synthetic validation needs no raw cohorts; real `prepare.py` needs the exact external input layout |

## Input validation failures

| Error/message | Meaning and correction |
| --- | --- |
| Missing explicitly weighted MLH1 | Provide the correctly calculated `weighted_MLH1`, or select a verified weighted alias |
| Missing/unmeasured fixed 18 features | Some required subtype columns are absent; recover supported measurements, not zero placeholders |
| Missing panel 12 features | Use all 12 named parent-normalized features; the old 10/7 panels are different models |
| Non-finite inputs | Locate NaN/Inf/blank/unmeasured values and correct their measurement provenance; no automatic imputation is performed |
| Fractions outside `[0,1]` | Do not submit percentages, whole-cell-normalized values, Z-scores or signed screening statistics |
| Duplicate/empty sample keys | scRNA requires unique `sampleID`; bulk requires unique `(dataset,sampleID)` if `dataset` exists |
| Wrong frozen/full-development model identity | Use the intended artifact. Bulk direct scoring requires panel 12/fold 0/seed 2026; fold 1–5 belong to OOF replay |
| Supplied LOCO lock belongs to another cohort | Use a genuine matching cohort/scale. An arbitrary lock does not define a new-cohort deployment model |
| Missing four pre-Z-score log columns | Add all four original-scale log2 MMR gene values for fusion, or run subtype-only mode |
| Invalid scaler/cutoff vector | Recover the actual JSON lock. Do not invent SDs, reorder genes or paste only a cutoff list |

### Why not fill missing MLH1 with zero?

`N_candidate=0` means no candidate malignant epithelial expression estimate. This is missing, even though the algebraic product would look like zero. Measured zero expression when candidate cells exist is a different event and is preserved. The direct scorer cannot reconstruct that biological distinction from a renamed column; prepare the feature correctly upstream.

### Why does a gene-expression CSV fail as a prediction input?

The direct predictors take subtype features, not a gene-only matrix. Bulk fractions require compatible deconvolution and within-parent normalization; scRNA requires fixed-reference cell measurement and weighted MLH1. The optional BayesPrism raw input orientation is genes×samples, while prediction CSV orientation is samples×features.

## Reproduction versus direct inference

**Bulk development results differ between commands:** `predict_bulk.py` uses the full-development model, so known development samples receive apparent scores. `reproduce.py --branch bulk` uses held-out-fold models for development OOF. Comparing them as if they were the same evaluation is incorrect.

**The scRNA scorer expects the saved feature scale:** an additional dataset-wise Z-score or a second `log1p` changes the input definition. Its StandardScaler is already frozen from training. MMR4 cohort standardization belongs to a separate branch and does not apply to the subtype predictors.

**MMR4 is high while subtype score is moderate:** the fixed fusion averages these distinct signals. The MMR4 sigmoid increases for expression below the fitted standardized thresholds, taking the maximum across four genes; subtype scoring is learned from ecological features.

**The bootstrap result does not change after replay:** the saved prediction/feature identity and exact 2026 resampling streams are intentionally reproduced. Bootstrap intervals are conditional on those fixed measurements and models, rather than a full redevelopment/feature-selection uncertainty analysis.

## Raw-data branch boundaries

The newer seed 2026 TOSICA dictionary supports 13/18 fixed-panel features, missing cTNI14,cTNI18,cTNI16,cTNI08,cTNI19. It cannot stand in for the older 18/18 ecological measurements. Correct checkpoint/reference provenance is required before deriving a new complete feature table.

Spatial replay rebuilds features from saved candidate allocation totals and posterior means; fresh reconstruction from original 10x counts is a different raw pipeline. Parameter means substituted into a reconstruction formula are not an average of per-draw reconstructed expression. Reference loss history was not recovered, so no convergence claim follows from the replay.

BayesPrism is a locally patched Python port. Its target updates use no MSI labels but are cohort-specific adaptations. The integer matrix is created before inference, whereas MMR4 continues to use the original log expression. Microarray linear intensity proxies do not become RNA-seq read counts through rounding.

## Clinical labels and score interpretation

Direct prediction does not need a label column. When retained as optional metadata, `MMR` is not used to estimate the score. Positive research coding combines MSI-H/dMMR source labels; not every public sample was confirmed by the same assay.

The examples and paper scores are not independently calibrated clinical probabilities. A value above 0.5 is the paper's fixed scoring convention, not a validated diagnostic cutoff for an arbitrary new population. Previously examined cohorts remain retrospective, and spatial proxy transfer does not establish measurement equivalence.
