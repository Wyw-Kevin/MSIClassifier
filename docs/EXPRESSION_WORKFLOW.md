# Expression data to MSIClassifier scores

[中文总览](../README_zh.md) · [English overview](../README.md) · [Feature table interfaces](INPUT_FILES.md)

The first biological input is gene expression. Subtype features are intermediate outputs produced by annotation or deconvolution. Use `predict.py` to run a new expression dataset. Use `reproduce.py` to replay archived paper analyses. Neither command trains or selects a new classifier.

## 1. Install code and the frozen reference resources

Unzip the code archive and enter its `MSIClassifier` directory. Use Python 3.12:

```bash
python -m pip install -r requirements.txt
# Additional dependencies for cell annotation, CNV and spatial mapping:
python -m pip install -r requirements-expression.txt
```

For `.loom` input also install its optional Scanpy reader dependency with `python -m pip install loompy`. This optional format was not part of the executed connection checks.

For bulk only, `requirements.txt` is sufficient. The existing patched BayesPrism port and its 72-type raw-count reference are already in the code archive. The original scRNA reference h5ad is not needed for this new bulk entry; it remains required only for rebuilding the reference with the archived `prepare.py`.

For scRNA/spatial, extract **both** `MSIClassifier_RawReference_part1.zip` and `MSIClassifier_RawReference_part2.zip` **inside the code directory**:

```bash
python -m zipfile -e /path/to/MSIClassifier_RawReference_part1.zip .
python -m zipfile -e /path/to/MSIClassifier_RawReference_part2.zip .
python predict.py --help
```

The resulting paths must be:

```text
MSIClassifier/
  predict.py
  resources/raw_reference/
    manifest.json
    ecology/tosica_major/
    ecology/CRC/{TNKILC,Myeloid,Strom,Plasma}/
    cnv_major/
    TOSICA_source/
    gene_positions.tsv
    spatial/{signature.csv,mapping.csv,complete.json,verification.json}
```

Each reference file is verified against its recorded SHA256 before use. The two reference archives together contain approximately 2.47 GB of uncompressed frozen resources. Do not put these large checkpoints in ordinary GitHub Git history; distribute the archive separately with the code release. No hosted download URL is assumed.

Linux is the tested platform. CPU is the default. `--device cuda:0` requests a working CUDA PyTorch installation; otherwise use `--device cpu`. Install a PyTorch build appropriate to your hardware. Expression mapping may need substantially more memory/time than scoring a feature table. Full cell matrices are loaded in memory.

Runnable bundled expression examples are in [examples/expression](../examples/expression/README.md). They start with gene matrices/h5ad rather than subtype features.

## 2. Bulk RNA-seq or processed microarray expression

### Required matrix and expression scale

CSV or TSV: **rows are gene symbols, columns are samples**, and the first column contains row identifiers:

```csv
gene,sampleA,sampleB
MLH1,7.1,6.5
MSH2,8.2,7.9
MSH6,7.6,8.0
PMS2,6.8,6.6
...```

Supply the complete measured gene matrix, not just four MMR genes or a small marker panel. Gene/sample identifiers must be unique and all values finite. At least 1,000 reference genes must overlap; the exact overlap is recorded. Do not precompute subtype proportions or classifier Z-scores.

Choose the scale explicitly:

| `--scale` | Supplied values | BayesPrism linear input | Pre-Z MMR4 values |
| --- | --- | --- | --- |
| `linear` | Nonnegative counts, normalized expression or linear intensities | Unchanged | `log2(x+1)` |
| `log2p1` | `log2(linear+1)` | `2**x-1` | Supplied values |
| `log2` | Processed log2 intensities | `2**x` | Supplied values |

Declare real platform units separately in your analysis records: linear intensities are not sequencing read counts. For example, the archived TCGA matrix uses `log2p1`, whereas the archived GEO processed log2 matrices use `log2`. Do not infer scale from the numerical range. VSN values, centered/scaled values and residuals cannot be inverted with these formulas.

```bash
python predict.py bulk \
  --expression my_bulk_expression.csv \
  --scale log2p1 \
  --dataset MY_COHORT \
  --output-dir output/my_bulk \
  --workers 4
```

The workflow:

1. Validate the supplied matrix and explicit scale.
2. Preserve continuous linear expression; apply nearest-even `np.rint` once to create the complete integer sampling matrix.
3. Run the bundled repaired Python BayesPrism implementation with its fixed 72-type raw-count reference. Each target cohort is processed independently, `key=None`, `update_gibbs=True`, chain 1,000, burn-in 500, thinning 2, alpha 1, seed 2026.
4. Read updated-reference posterior mean fractions `posterior_theta_f.theta`; divide each of the selected 12 subtypes by its parent total.
5. Apply the unchanged full-development panel12 scaler and MLP.

This is target-cohort unsupervised adaptation. Supplying only part of a cohort may change its reference update and estimates.

### Microarray probe matrix

For a probe-level processed matrix, provide a separate CSV containing `probe_id,gene_symbol`:

```bash
python predict.py bulk \
  --expression processed_probe_expression.tsv \
  --probe-map probe_to_gene.csv \
  --scale log2 \
  --dataset MY_ARRAY_COHORT \
  --output-dir output/my_array
```

Unmapped probes are excluded. Each mapped probe must have one unambiguous gene symbol. Multiple probes for the same gene are combined by median **on the supplied scale**, then transformed to linear expression. This explicit portable aggregation is not a claim that it reproduces every historical platform's gene-summary method.

Raw CEL/IDAT/scanner files must first undergo platform-appropriate background correction/normalization and probe annotation. This tool does not parse such files or implement one common preprocessing method for all platforms. Supply the processed gene or probe expression matrix above.

### Optional matching-cohort MMR4 fusion

```bash
python predict.py bulk \
  --expression GSE24550_log2_expression.csv \
  --scale log2 \
  --dataset GSE24550 \
  --mmr4-lock workflows/bulk/mmr4/locks/heldout_GSE24550.json \
  --output-dir output/GSE24550_fresh
```

This applies the existing cohort's archived pre-Z scaler/thresholds, then fixed `0.5*Subtype+0.5*MMR4_LOCO`. These round-specific locks do not provide an arbitrary new cohort with deployment calibration. New datasets default to the subtype score. Never rename an unknown cohort to make a lock pass. Four MMR gene values are taken from the unrounded log expression; they do not come from the BayesPrism sampling matrix.

## 3. Single-cell expression

### Accepted files

- `.h5ad`: choose `X` or an explicit `--layer counts`.
- 10x `.h5`: `--format 10x-h5` (also recognized automatically).
- A 10x Matrix Market directory: `--format 10x-mtx`, containing matrix, features/genes and barcodes files.
- `.loom`: `--format loom`.
- CSV/TSV: default rows are cells and columns are gene symbols; add `--transpose` for gene-by-cell tables.

An arbitrary `.h5` is not necessarily a 10x HDF5 file. For another Scanpy-readable format, read it with its appropriate reader, attach cell/gene IDs and save `.h5ad`. For a standalone MTX, attach its gene/cell identifiers in AnnData and save h5ad, or use a complete 10x MTX directory.

Gene symbols must be unique. If `var_names` are Ensembl IDs, provide `--gene-symbol-column gene_symbol` when that `var` column is available. Resolve duplicate symbols before normalization; do not sum log expression values.

Each cell must have a specimen identifier `Sample` and tissue designation `Class` (`Tumor` for tumor samples). They may be in `adata.obs` or an external CSV:

```csv
cell_id,Sample,Class
AAAC-1,tumorA,Tumor
AAAG-1,tumorA,Tumor
AAAT-1,normalA,Normal
```

These fields identify samples and tissue origin; **MSI/MSS labels are not required**. External metadata must uniquely cover every input cell. Use `--sample-column`, `--tissue-column` and `--tumor-value` for other column/value names.

Raw counts:

```bash
python predict.py scrna \
  --expression my_cells.h5ad \
  --layer counts \
  --scale counts \
  --dataset MY_SCRNA \
  --output-dir output/my_scRNA \
  --device cpu
```

10x HDF5 with separate metadata:

```bash
python predict.py scrna \
  --expression filtered_feature_bc_matrix.h5 \
  --metadata cell_metadata.csv \
  --scale counts \
  --dataset MY_SCRNA \
  --output-dir output/my_10x \
  --device cuda:0
```

For already normalized expression, replace `--scale counts` with `--scale log1p-cp10k` and select the corresponding layer or X. This declaration requires natural-log `log1p` of full-library CP10k expression; HVG-only, scaled, residual, log2 and doubly normalized matrices are incompatible. Library normalization occurs **before** genomic/model-gene subsetting. A genuine raw matrix must have nonnegative integer values and no zero-library cells.

The workflow retains samples with at least 200 total cells, independently applies the frozen historical ecology Major/CRC models with exact gene orders, and separately uses the current seed2026 Major model for Strom/Epi calls in the CNV branch. CNV uses Strom references only, a unique union of at most 100,000 tumor-tissue cells and 20,000 reference cells, and the reference q95 score rule. It calculates the 18 within-parent proportions and weighted MLH1 from the same retained CNV epithelial/candidate set, then applies the current frozen scaler/LDA.

The ecology family supports all 18 required classifier types; it does not measure all 71 candidate subtypes. The separate newer full hierarchical family supports 13/18, so it is not silently substituted. Historical checkpoint training seeds remain their actual values (Major 20260723; some CRC seeds unknown), while all new explicit RNG settings are 2026. Missing model genes are zero-aligned following the historical annotation implementation and reported; an unsupported subtype is never zero-filled.

For `--dataset GSE178341`, original `clTopLevel` and `cl295v11SubShort` must be supplied and are used as the fixed reference definitions.

If a sample lacks a required parent denominator or candidate malignant cells, its feature is missing. Such samples are listed as `unmeasured_feature_abstain` and are not assigned a model score. Real zero MLH1 with candidate cells is retained. No 50-cell malignant eligibility cutoff is applied.

## 4. Spatial expression

Use **unnormalized nonnegative integer spot counts**, with one section per invocation. The same file readers as scRNA are available. For h5ad with log-normalized X and raw counts in a layer, explicitly use `--layer counts`.

```bash
python predict.py spatial \
  --expression my_section.h5ad \
  --layer counts \
  --sample-id sectionA \
  --output-dir output/sectionA \
  --device cuda:0
```

Or supply the 10x HDF5 and optionally spot coordinates:

```bash
python predict.py spatial \
  --expression filtered_feature_bc_matrix.h5 \
  --positions spot_coordinates.csv \
  --sample-id sectionA \
  --output-dir output/sectionA
```

The optional positions CSV has index `spot_id` and columns `x,y`, uniquely covering every spot. Existing `obsm["spatial"]` is retained. Coordinates are useful for mapping displays; cell2location abundance fitting and section-level scoring do not require them.

The workflow uses the saved current 73-type regression signature (71 non-Epi types plus `Epi_CNV_candidate` and `Epi_other`), not the legacy spatial API. It fits a new section with the recorded budget: 500 epochs, 200 posterior draws, `N_cells_per_location=8`, `detection_alpha=20`, seed2026. It requires MLH1 and at least2,000 shared signature genes. It does not refit the reference regression.

Candidate counts are allocated using the official conditional-expression formula evaluated at saved posterior means, checked against the official implementation on the first spots. Counts are normalized over **all shared retained signature genes** to CP10k and log1p. Section weighted MLH1 is:

```text
sum_spots(candidate_MLH1_log1p_CP10k * candidate_abundance)
 / sum_spots(candidate_abundance + other_epithelial_abundance)
```

The 18 reconstructed subtype/parent ratios and this proxy are passed to the frozen current scRNA classifier. `Epi_other` is not confirmed normal epithelium. Reference epoch history was not recovered; a fixed section budget does not establish convergence. This is exploratory proxy transfer, not evidence of cross-platform measurement equivalence.

## 5. Inspect files before running models

Add `--check-only` to any command above, using a new output directory. It writes `input_check.json` and exits before reference loading, annotation/deconvolution or scoring. It checks matrix orientation/IDs/values, explicit scale, and required specimen metadata. It does not certify reference coverage or model execution.

All execution commands require a new or empty output directory. This prevents stale cached posterior tasks or prior outputs from being reused with another dataset.

## 6. Outputs and unchanged advanced entries

| Output | Content |
| --- | --- |
| `input_check.json` | Input format, dimensions, scale and preflight scope |
| `features.csv` | Complete sample/section classifier inputs |
| `scores.csv` | Frozen classifier score; optional matching MMR4/fusion |
| `run_manifest.json` | Actual software versions, reference identities and processing settings |
| Bulk `final_fractions.csv` | Final BayesPrism72-type posterior mean abundances |
| Bulk `linear_expression.pkl`, `sampling_expression.pkl` | Continuous and rounded sample-by-gene inputs |
| scRNA `annotations.csv.gz`, `cnv_retained_cells.csv.gz` | Cell identity, retained CNV cells and candidate calls |
| scRNA `all_samples_features.csv` | Complete/incomplete features and abstention status |
| Spatial `abundance.csv`, `spot_proxy.csv.gz` | Posterior abundance and per-spot conditional proxy |
| Spatial `cell2location_model/`, `posterior_means.npz`, `history_*.csv` | New fitted section state, posterior means and losses |

Already measured compatible features may still be scored with `workflows/scRNA/predict_scrna.py` and `workflows/bulk/predict_bulk.py`. The archived-input `reproduce.py --branch all` remains unchanged. These are shortcuts for completed upstream work rather than the first required input format.

New expression runs do not automatically reproduce archived cohort performance: annotation identity, cohort adaptation, sampled CNV cells, shared genes, hardware and versions can affect measurements. No new AUROC is claimed by input/connection checks.

Reader and model interfaces: [Scanpy read_h5ad](https://scanpy.readthedocs.io/en/stable/generated/scanpy.read_h5ad.html), [Scanpy read_10x_h5](https://scanpy.readthedocs.io/en/stable/generated/scanpy.read_10x_h5.html), [infercnvpy score definition](https://infercnvpy.readthedocs.io/en/latest/generated/infercnvpy.tl.cnv_score.html), [cell2location raw-count and reference workflow](https://cell2location.readthedocs.io/en/latest/notebooks/cell2location_tutorial.html).
