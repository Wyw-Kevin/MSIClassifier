# MSIClassifier

**Colorectal cancer MSI/dMMR research scoring from cell subtype composition and MMR expression.**

MSIClassifier connects single-cell, bulk and spatial transcriptomic analyses through fixed cell subtype definitions. Its classifiers combine cell composition with MMR-associated expression information. The main prediction entry starts from bulk gene/probe expression or single-cell/spatial expression files and constructs the required subtype and MMR features internally. BAM files and MSI outcome labels are not required.

Three entry points are provided: expression-to-score prediction, direct scoring of already measured features, and replay of the paper's stored-input analyses. The required large frozen annotation resources are supplied as a separate reference archive; original full raw datasets and manuscript supplements remain separate. The repository is used through Python scripts; it is not currently a PyPI command or Docker image. [中文说明](README_zh.md).

## Contents

- [Models and scope](#models-and-scope)
- [Installation](#installation)
- [Start from expression data](#start-from-expression-data)
- [Already measured features](#already-measured-features)
- [Input files](#input-files)
- [Command options](#command-options)
- [Reproducing analyses](#reproducing-analyses)
- [Outputs](#outputs)
- [Detailed documentation](#detailed-documentation)
- [License](#license)

## Models and scope

| Workflow | Intermediate/classifier input | Supplied model | Output |
| --- | --- | --- | --- |
| scRNA | 18 subtype proportions and malignant-fraction weighted MLH1 | Training StandardScaler and shrinkage LDA | Specimen-level `score` |
| Bulk subtype | 12 parent-normalized subtype abundances | Full-development StandardScaler and 4-unit MLP | Sample-level `Subtype` |
| Existing-cohort bulk fusion | Bulk subtype features, four log2 gene values and a matching MMR4 lock | Full-development MLP plus saved cohort MMR4 parameters | `Subtype`, `MMR4_LOCO`, `Fusion05` |
| Spatial expression | Raw spot counts, saved73-type reference, reconstructed18+weightedMLH1 proxy | Frozen scRNA classifier | New section score |
| Spatial replay | Bundled allocated-count, abundance and posterior-mean records | Frozen scRNA classifier applied to reconstructed proxies | 14 section and 7 patient scores |

Weighted MLH1 is:

~~~text
weighted_MLH1 = mean(candidate malignant epithelial MLH1)
                × N_candidate_malignant / N_all_epithelial
~~~

Both counts come from the same retained CNV cell set. Cell-level MLH1 uses total-count normalization to 10,000 and natural-log `log1p` before averaging candidate cells. Zero candidate cells means missing; measured zero expression with eligible candidate cells is retained. Whole-epithelial mean MLH1 does not replace this feature.

Bulk fusion is `Fusion05 = 0.5 × Subtype + 0.5 × MMR4_LOCO`. The eight MMR4 locks are eight held-out-cohort evaluation artifacts, not one universal deployment model. A fusion input must match its lock's cohort. New cohorts can use subtype scoring; they cannot select an arbitrary evaluation-round lock.

Scores increase toward the positive research outcome, combining public MSI-H or IHC-dMMR labels. They are continuous model outputs rather than clinically calibrated probabilities.

## Installation

### Requirements

- Python **3.12**; the audited environment used 3.12.12.
- Linux is the tested platform. BayesPrism uses Linux multiprocessing/file locking; spatial plots also need fontconfig.
- Feature-table scoring and bundled-input replay need no GPU. The new cell/spot mapping routes default to CPU and can request CUDA explicitly.
- Keep supplied models and resource folders with the scripts. Exact tested dependencies are pinned in [requirements.txt](requirements.txt).

Extract/download the repository and enter its `MSIClassifier` directory. All commands below assume that working directory.

~~~bash
cd MSIClassifier
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
~~~

Alternatively:

~~~bash
conda create -n msiclassifier python=3.12 pip
conda activate msiclassifier
python -m pip install -r requirements.txt
~~~

For paired spatial plots on Ubuntu/Debian, install fontconfig if `fc-match` is missing:

~~~bash
sudo apt-get install fontconfig
~~~

Inspect interfaces without loading models:

~~~bash
python predict.py --help
python predict.py bulk --help
python predict.py scrna --help
python predict.py spatial --help
python reproduce.py --help
python workflows/scRNA/predict_scrna.py --help
python workflows/bulk/predict_bulk.py --help
~~~

BayesPrism imports the included `workflows/bayesprism/vendor/` implementation. Installing another port does not substitute for this preserved patched version.

## Start from expression data

Install `requirements.txt`; scRNA/spatial additionally use `requirements-expression.txt` and **both** `MSIClassifier_RawReference_part1.zip` and `MSIClassifier_RawReference_part2.zip`. Extract the resource archive inside this directory so that `resources/raw_reference/manifest.json` exists. Bulk uses its already bundled BayesPrism reference.

```bash
# Bulk: gene rows × sample columns, explicit scale
python predict.py bulk --expression bulk_expression.csv --scale log2p1 \
  --dataset MY_BULK --output-dir output/my_bulk

# Single-cell: raw counts in a h5ad layer
python predict.py scrna --expression cells.h5ad --layer counts --scale counts \
  --dataset MY_SCRNA --output-dir output/my_scRNA

# Spatial: one raw-count section
python predict.py spatial --expression section.h5ad --layer counts \
  --sample-id sectionA --output-dir output/sectionA
```

Single-cell input requires specimen and tumor/normal tissue metadata in obs or `--metadata`. Readers support h5ad, 10x HDF5, 10x MTX directories, loom and cell/spot-by-gene text. Add `--transpose` for gene-by-cell text. An arbitrary HDF5 file is not necessarily a 10x file. Microarray probe matrices additionally need `--probe-map`; raw CEL/IDAT requires platform preprocessing first.

Each route writes measured/reconstructed `features.csv`, `scores.csv`, and a processing manifest. Add `--check-only` to inspect inputs without running models. Use a new output directory. These are new technical runs with unchanged classifier weights, not an automatic recomputation of archived AUROC.

[Runnable bundled expression examples](examples/expression/README.md) · [Complete expression workflow, required files and resource installation](docs/EXPRESSION_WORKFLOW.md) · [All expression parameters](docs/EXPRESSION_PARAMETERS.md)

## Already measured features

### Score the scRNA feature example

~~~bash
python workflows/scRNA/predict_scrna.py \
  --feature-table examples/scRNA_features.csv \
  --output output/scRNA_scores.csv
~~~

Five specimens are supplied with all 19 numerical features. Output is `sampleID,score`. The first two expected scores are approximately 0.119320418 and 0.083181782. Compare against [examples/scRNA_expected_scores.csv](examples/scRNA_expected_scores.csv).

### Score the bulk feature example

~~~bash
python workflows/bulk/predict_bulk.py \
  --feature-table examples/bulk_features.csv \
  --output output/bulk_scores.csv
~~~

This uses the full-development MLP without requiring labels or MMR4. Reference output: [examples/bulk_expected_scores.csv](examples/bulk_expected_scores.csv).

### Apply the matching GSE24550 fusion lock

~~~bash
python workflows/bulk/predict_bulk.py \
  --feature-table examples/bulk_features.csv \
  --mmr4-lock workflows/bulk/mmr4/locks/heldout_GSE24550.json \
  --output output/bulk_fusion.csv
~~~

These example rows belong to GSE24550 and include four log2 gene values. Output is `dataset,sampleID,Subtype,MMR4_LOCO,Fusion05`. The saved cohort scaler is used, not a scaler estimated from the five example rows. Reference output: [examples/bulk_expected_fusion.csv](examples/bulk_expected_fusion.csv).

### 4. Use your own samples

Keep exact required column names/measurement definitions and replace `--feature-table` with your CSV path. Supply unscaled fractions and the verified weighted MLH1 feature; the model applies its training scaler internally. **Do not Z-score again**, fill unmeasured types with zero or substitute raw/whole-epithelial MLH1. No `y` or MSI-status column is needed. [Detailed tutorial](docs/TUTORIAL.md).

## Input files

All direct-scoring files are headered comma-separated tables with **samples as rows** and **features as columns**. Numerical features must be finite. Additional metadata columns are allowed.

| Command | Required columns | Optional columns |
| --- | --- | --- |
| `predict_scrna.py` | Unique `sampleID`, all 18 subtype columns, `weighted_MLH1` | `dataset`, `MMR`; explicit weighted-column alias |
| `predict_bulk.py` | Unique sample keys, all 12 subtype columns | `dataset` |
| Bulk with `--mmr4-lock` | Above plus matching `dataset` and four `*_raw` gene columns | Other metadata |

**scRNA subtype columns:**

~~~text
cTNI14, cTNI18, cS18, cTNI13, cTNI16, cTNI05, cTNI07, cS27,
cTNI04, cS17, cM01, cP1, cM05, cM06, cTNI08, cTNI19, cTNI10, cM04
~~~

**Bulk subtype columns:**

~~~text
cS27, cTNI16, cM06, cTNI05, cTNI10, cP1,
cTNI13, cS18, cTNI04, cS17, cM05, cTNI08
~~~

Subtype values are within-parent fractions, not percentages of all cells: cTNI uses TNKILC, cS uses Strom, cM uses Myeloid and cP1 uses Plasma. Use 0–1, not 0–100. Exact mappings: [examples/required_features.csv](examples/required_features.csv).

MMR4 requires `MLH1_raw`, `MSH2_raw`, `MSH6_raw` and `PMS2_raw`. Despite their names, these are **pre-Z-score log2** values: TCGA `log2(RSEM+1)` or processed GEO log2 microarray expression. They are not BayesPrism's inverse-transformed/rounded inputs. [Complete schemas and file inventory](docs/INPUT_FILES.md).

## Command options

### scRNA scoring

| Option | Required/default | Purpose |
| --- | --- | --- |
| `--feature-table` | Required | Input sample-by-feature CSV |
| `--output` | Required | Output CSV; parent directories are created |
| `--model` | Included frozen scRNA artifact | Explicit compatible model path |
| `--weighted-mlh1-column` | `weighted_MLH1` | Name of the verified weighted feature column |
| `-h`, `--help` | Flag | Command help |

### Bulk scoring

| Option | Required/default | Purpose |
| --- | --- | --- |
| `--feature-table` | Required | Input sample-by-feature CSV |
| `--output` | Required | Output CSV |
| `--model` | Included `panel12_fold0.joblib` | Full-development 12-feature artifact |
| `--mmr4-lock` | Omitted | Enable MMR4/fixed fusion with a matching saved lock |
| `-h`, `--help` | Flag | Command help |

### Analysis dispatcher

| Option | Default | Purpose |
| --- | --- | --- |
| `--branch` | `all` | `all`, `bulk`, `scRNA`, `sensitivity`, `spatial`, `paired`, `stability`, `bayesprism` or `plots` |
| `--output-dir` | Repository `output/` | Separate result/log directory |
| `--plots` | Off | Add numerical plots; spatial replay also adds paired spot plots |

Panels, stored scalers, fixed fusion weights and MMR4 temperature are model settings, not CLI tuning options. [All command/model/environment parameters](docs/PARAMETERS.md).

## Reproducing analyses

~~~bash
# Stored-input replay and verification; no refitting
python reproduce.py --branch all

# Include numerical figures
python reproduce.py --branch all --plots

# Individual comparisons
python reproduce.py --branch paired
python reproduce.py --branch stability
~~~

Bulk development replay uses the stored five-fold models; direct bulk scoring uses the full-development model. Optional refitting and real-cohort BayesPrism execution are separate [tutorial](docs/TUTORIAL.md) commands, excluded from `--branch all`.

## Outputs

| File | Contents |
| --- | --- |
| Direct scRNA output | Identifiers and continuous `score` |
| Direct bulk output | Identifiers and `Subtype`; with a lock, also `MMR4_LOCO`/`Fusion05` |
| `output/bulk/predictions.csv` | OOF/retrospective score replay |
| `output/bulk/macro_metrics.csv` | Equal-cohort macro AUROC |
| `output/scRNA/replay_verification.json` | Frozen scores, patient bootstrap and control verification |
| `output/spatial/reproduced_patient_predictions.csv` | Mean of two section scores per patient |
| `output/paired/S13_fusion_vs_MMR4_macro.csv` | Paired fusion-minus-MMR4 differences/95% intervals |
| `output/stability/S14.csv` | All 17 candidates' selection frequencies |
| `output/summary.json` | Dispatcher completion summary |

For the supplied expected outputs, compare numeric values with tolerance around `1e-10` in the pinned environment.

## Detailed documentation

- [Tutorial: expression inputs followed by advanced examples, refitting and replay](docs/TUTORIAL.md)
- [Input schemas and bundled/external file inventory](docs/INPUT_FILES.md)
- [Command/model parameter reference](docs/PARAMETERS.md)
- [Troubleshooting and interpretation](docs/TROUBLESHOOTING.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

Results condition on fixed upstream features and previously viewed cohorts. Older ecological TOSICA weights support 18/18 classifier types; the newer seed 2026 family supports 13/18. The expression scRNA route explicitly uses the older ecology family and the newer Major-only CNV reference. Spatial reconstruction is proxy-feature transfer, and missing reference epoch history does not establish convergence. Detailed documentation preserves these boundaries.

## License

Original MSIClassifier software and documentation are distributed under the **MIT License**. See [LICENSE](LICENSE) for the full terms. Bundled third-party software retains its own license, and original data conditions remain applicable.
