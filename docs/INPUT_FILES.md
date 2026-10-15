# Input formats and required files

Start from bulk gene/probe expression or scRNA/spatial expression files: [Expression input schemas](EXPRESSION_WORKFLOW.md). The existing tables below describe stored-feature scoring and archived replay where stated.

[Main README](../README.md) · [Tutorial](TUTORIAL.md) · [Parameters](PARAMETERS.md)

## 1. General CSV conventions

Direct scoring reads a headered comma-separated file, with one sample per row. Column names are case-sensitive: use `sampleID`, not `SampleID`. Numeric features must be finite; blank/NaN/Inf inputs cannot replace an unmeasured feature. Fractions are decimal values in `[0,1]`, not percentages. Do not log-transform or Z-score these fractions.

Column order can differ from the examples: each scorer selects columns in the frozen model's feature order. Additional metadata is ignored as a model input. Labels are not required. Extra `MMR` metadata in an scRNA table is preserved in its output but does not influence scoring.

## 2. scRNA scoring CSV

Required files are your feature CSV, `workflows/scRNA/predict_scrna.py`, and the supplied classifier at `workflows/scRNA/resources/model/reference/auc070_17_weighted_mlh1_2026/model.joblib`. The default model path is resolved relative to the script, so keeping the repository layout is sufficient.

| Field | Type | Definition |
| --- | --- | --- |
| `sampleID` | Nonempty unique identifier | One tumor specimen per row; repeat patient specimens retain distinct sample IDs |
| 17 subtype columns below | Finite decimal fractions | Subtype cell count divided by its major parent count, using the frozen reference/QC definitions |
| `weighted_MLH1` | Finite continuous value | Candidate malignant epithelial MLH1 mean multiplied by the candidate fraction within all retained epithelial cells |
| `dataset` | Optional identifier | Metadata only |
| `MMR` | Optional text | Metadata only; not needed for prediction |

### Exact subtype names and denominators

| Parent | scRNA features |
| --- | --- |
| TNKILC | `cTNI14`, `cTNI18`, `cTNI13`, `cTNI16`, `cTNI05`, `cTNI07`, `cTNI04`, `cTNI08`, `cTNI19`, `cTNI10` |
| Strom | `cS18`, `cS27`, `cS17` |
| Myeloid | `cM01`, `cM05`, `cM06` |
| Plasma | `cP1` |

For example, `cTNI16 = N_cTNI16 / N_TNKILC`, not `N_cTNI16 / N_all_cells`. The archived ecology measurements included a zero-encoding convention for some absent major classes. Distinguish a measured biological absence under that convention from an unsupported/unmeasured subtype; the latter must not be filled with zero. The newer seed 2026 annotation family lacks five fixed-panel types and cannot replace the older measurements.

The complete header is:

```text
sampleID,cTNI14,cTNI18,cS18,cTNI13,cTNI16,cTNI05,cTNI07,cS27,cTNI04,cS17,cM01,cP1,cM05,cM06,cTNI08,cTNI19,cTNI10,weighted_MLH1
```

See [the complete five-row example](../examples/scRNA_features.csv). A small excerpt of the same values is:

| sampleID | cTNI14 | cS27 | cM01 | cP1 | weighted_MLH1 |
| --- | --- | --- | --- | --- | --- |
| SMC01-T | 0.102894 | 0.142857 | 0.435294 | 0.075758 | 0.050373 |
| SMC02-T | 0.081594 | 0.169811 | 0.663961 | 0.050000 | 0.063630 |

This excerpt omits other required columns for readability; use the complete CSV as the input template.

### MLH1 measurement contract

1. Use the reference/QC-compatible cell annotation and per-cohort stromal-only CNV measurement.
2. Candidate malignant epithelial cells meet the study-defined stromal reference q 95 rule. This is not an official infercnvpy malignancy threshold.
3. Calculate candidate MLH1 mean from the expression layer corresponding to per-cell total-count 10,000 normalization and natural-log `log1p`, before CNV gene subsetting.
4. Multiply by `N_candidate / N_all_epithelial`, with both counts from the same retained CNV cell set.

For a mean of `0.5`, 20 candidate cells and 80 retained epithelial cells, `weighted_MLH1 = 0.5 × 20/80 = 0.125`. No per-sample 50-cell threshold applies. Zero candidates or a zero epithelial denominator means missing; measured zero MLH1 among eligible candidate cells remains zero.

The current artifact uses `weighted_MLH1`. Some historical weighted tables use the slot `mean_MLH1_epithelial`. If you supply such a verified table, select it explicitly with `--weighted-mlh1-column mean_MLH1_epithelial`. The same column name in an older raw/whole-epithelial table does not make it compatible.

## 3. Bulk subtype scoring CSV

Required files are your CSV, `workflows/bulk/predict_bulk.py`, and `workflows/bulk/classifier_snapshot/models/panel12_fold0.joblib`.

| Field | Requirement |
| --- | --- |
| `sampleID` | Nonempty sample identifier |
| `dataset` | Optional for subtype-only scoring; if present, `(dataset,sampleID)` must be unique |
| Twelve columns below | Required finite parent-normalized fractions in `[0,1]` |

| Parent | Bulk features |
| --- | --- |
| TNKILC | `cTNI16`, `cTNI05`, `cTNI10`, `cTNI13`, `cTNI04`, `cTNI08` |
| Strom | `cS27`, `cS18`, `cS17` |
| Myeloid | `cM06`, `cM05` |
| Plasma | `cP1` |

Each denominator sums **all reference subtypes of that parent**, not only the selected 12 features. Thus the 12 columns do not need to sum to one. Do not multiply inputs by a subtype's screening-direction sign; direction is used in selection, while the classifier takes the original nonnegative fractions.

```text
dataset,sampleID,cS27,cTNI16,cM06,cTNI05,cTNI10,cP1,cTNI13,cS18,cTNI04,cS17,cM05,cTNI08
```

The supplied [bulk example](../examples/bulk_features.csv) also contains the four MMR4 columns, which subtype-only scoring ignores.

## 4. MMR4 fusion: extra inputs and lock

Fusion additionally needs a matching dataset-specific JSON lock and these columns:

| Column | Expression scale before saved cohort standardization |
| --- | --- |
| `MLH1_raw` | Stored log2 MLH1 expression |
| `MSH2_raw` | Stored log2 MSH2 expression |
| `MSH6_raw` | Stored log2 MSH6 expression |
| `PMS2_raw` | Stored log2 PMS2 expression |

TCGA uses `log2(RSEM normalized expression+1)`; the seven GEO datasets use processed log2 microarray values. Do not provide already Z-scored values, linear inverse transforms, integer-rounded BayesPrism input or TPM values on another scale.

Locks are `workflows/bulk/mmr4/locks/heldout_<DATASET>.json`. Each contains:

| JSON key | Purpose |
| --- | --- |
| `held_out_cohort` | Dataset to which this evaluation lock applies |
| `training_cohorts` | Seven complement cohorts used for threshold fitting |
| `cutoffs` | Four thresholds in MLH1/MSH2/MSH6/PMS2 order |
| `held_out_scaler.mean` | Four saved cohort means |
| `held_out_scaler.sample_sd` | Four saved sample SDs, `ddof=1` |
| `held_out_labels_used_for_fit` | Must be false |

The direct fusion scorer uses the stored means/SDs. It does not recalculate them on a smaller input subset. Renaming a new cohort to an existing `dataset` does not establish measurement compatibility. There is no supplied universal new-cohort deployment lock.

## 5. Included files versus additional raw resources

| Task | Included resources | Additional resources required |
| --- | --- | --- |
| Direct scRNA scoring | Frozen LDA/scaler and five-row feature example | Your complete compatible feature table |
| Direct bulk subtype scoring | Full-development MLP/scaler and five-row example | Your 12 parent-normalized features |
| Matching-cohort fusion | Eight MMR4 locks and example log values | Compatible four-gene values from that cohort |
| Stored-input bulk/scRNA analyses | Evaluation features, labels, folds, scores, bootstrap records and required saved models | None beyond software dependencies |
| Spatial saved-input replay | 14-section spot allocations/abundances/posterior means, features, comparator and scores | None beyond software dependencies |
| Fresh BayesPrism preparation/inference | Actual vendor, cell/sample manifests and counts-reference summary | Original reference h 5ad and target expression matrices |
| Fresh raw scRNA annotation/CNV | Definitions/provenance and complete source in separate full archive | Raw h 5ad, actual older frozen TOSICA checkpoints and raw-pipeline dependencies |
| Fresh raw spatial reconstruction | Complete source/provenance in separate full archive | Reference h 5ad, 10x matrices and cell 2location dependencies/checkpoints |

Only the first five tasks are immediately executable from the compact repository. Its default replay does not run raw TOSICA, CNV, cell 2location fitting or real-cohort BayesPrism inference.

## 6. BayesPrism raw-input layout

Set `MSI_PROJECT_ROOT` to a separately prepared directory:

```text
input_project/
├── GSE178341.h5ad
└── bulk_data/
    ├── TCGA-COADREAD/expression.csv
    ├── GSE39582/expression.csv
    ├── GSE24550/expression.csv
    ├── GSE25071/expression.csv
    ├── GSE27544/expression.csv
    ├── GSE35896/expression.csv
    ├── GSE39084/expression.csv
    └── GSE75316/expression.csv
```

The reference h 5ad needs cell IDs matching the supplied cell manifest and raw `layers['counts']`; the preparation script groups the recorded 72 reference types and aggregates repeated gene symbols. Target `expression.csv` files use **genes as rows, sample IDs as columns**, with the first column holding unique gene symbols. Sample columns must include the selected IDs from `resources/reference/sample_manifest.csv`. These matrices are the archived gene-level representations, not arbitrary current downloads of a GEO series.

Reference files under `workflows/bayesprism/resources/reference/`:

| File | Content |
| --- | --- |
| `cell_manifest.csv.gz` | Indexed cell IDs, sample/QC labels and fixed `CellType` |
| `sample_manifest.csv` | Selected `dataset,sampleID` pairs |
| `mapping.csv` | Reference `CellType,n_cells,parent` |
| `raw_count_profiles.pkl` | Preaggregated raw-count profiles, 72 types × 17,039 genes |
| `gene_audit.csv` | Reference gene filtering records |

The current `prepare.py` rebuilds the reference from the original h 5ad; the bundled count summary alone does not make that command independent of the h 5ad. Target preparation records/removes genes with non-finite values, checks remaining values, inverses the logged scale, stores continuous matrices and rounds the full sampling matrix with nearest-even `np.rint`. Negative inputs or invalid transformed values stop execution.

## 7. Spatial saved-input files

`workflows/spatial/resources/transfer/` contains `analysis_plan.json`, `reference/mapping.csv` and 14 `VISDS...` section folders. Each section folder has:

| File | Content used by replay |
| --- | --- |
| `small_cell_abundance_means.csv` | Posterior mean abundance by reference type and spot |
| `spot_mlh1_and_abundance.csv.gz` | Coordinates, candidate counts/library totals and reconstructed log expression |
| `posterior_means.npz` | Saved parameter means; `w_sf` is checked against abundance |
| `history_elbo_train.csv` | 500 recorded section-loss entries |

`resources/report/` holds the frozen classifier feature table, section/patient predictions, descriptive AUROC and ecology-only comparator. The main LDA is shared from `workflows/scRNA/`. The spot records permit reconstruction from saved allocation totals; they do not replace original 10x counts for fresh conditional allocation or posterior sampling.
