# MSIClassifier

Executable research code for the colorectal cancer MSI/dMMR transcriptomic classifier. The repository contains working scientific entrypoints, their dependencies, compact example/evaluation inputs, frozen sample-level classifiers and parameter locks. Manuscript documents, supplementary workbooks, generated figures, raw h5ad/10x data and historical exploratory pipelines are provided separately.

This local repository has not yet been published to GitHub. It preserves the manuscript's actual model identities; packaging does not retrain or validate a new model. [Chinese README](README_zh.md).

## Install and reproduce

Use Python3.12 in a separate environment:

```sh
python -m pip install -r requirements.txt
python reproduce.py --branch all
```

The default runner performs archived-input CPU replay, sensitivity-model verification, paired/bootstrap analyses and synthetic BayesPrism tests. It performs no raw preprocessing or model fitting. Outputs go to `output/`. Choose a single branch with `--branch bulk`, `scRNA`, `sensitivity`, `spatial`, `paired`, `stability`, `bayesprism`, or `plots`. `--branch all --plots` also regenerates scientific figures; optional spatial plot rendering requires Linux fontconfig.

## Direct scoring without outcome labels

### scRNA

```sh
python workflows/scRNA/predict_scrna.py --feature-table examples/scRNA_features.csv --output output/scRNA_scores.csv
```

Input requires `sampleID`, the18 model subtype columns, and `weighted_MLH1`. The last variable must equal mean(candidate malignant epithelial MLH1)×N_candidate/N_all_epithelial using counts from the same retained CNV set. The frozen training scaler/LDA is applied directly. Missing/unmeasured features stop scoring. This command does not annotate raw cells or estimate CNV. Archived `mean_MLH1_epithelial` columns represent weighted MLH1 only when their measurement provenance explicitly establishes that; pass `--weighted-mlh1-column` to use that alias.

### Bulk subtype branch

```sh
python workflows/bulk/predict_bulk.py --feature-table examples/bulk_features.csv --output output/bulk_scores.csv
```

Input requires unique sample keys and the12 subtype abundances divided by the total abundance of their respective parent cell classes. The command applies the actual full-development panel12 MLP and training scaler. It does not require labels and does not select features or train a model. Development scores from this full-fit command are apparent predictions; use `reproduce.py --branch bulk` for the paper's archived five-fold OOF predictions.

### Explicit existing-cohort MMR4 fusion

```sh
python workflows/bulk/predict_bulk.py --feature-table examples/bulk_features.csv --mmr4-lock workflows/bulk/mmr4/locks/heldout_GSE24550.json --output output/bulk_fusion.csv
```

This example contains GSE24550 rows. It uses that cohort's saved cutoff and scaler lock, four `*_raw` **log2 expression** columns, and fixed0.5/0.5 fusion. The eight LOCO locks are distinct evaluation-round artifacts, **not one universal deployment model**. Locks for unmatched/new cohorts are rejected. A new cohort can use the subtype command; deployment of an MMR4/fusion model needs a separately defined lock and cohort normalization protocol, not selection of a convenient LOCO round.

## Working analysis code

| Directory | Executable content and scope |
| --- | --- |
| `workflows/bulk/` | Direct scoring, actual fold-checkpoint replay, current7/12 feature refit, MMR4 optimizer, paired increment and development selection stability |
| `workflows/scRNA/` | Direct frozen scoring, primary/controls replay, patient bootstrap, rho/VIF and current uncapped subtype/CNV sensitivity |
| `workflows/spatial/` |14-section/7-patient frozen transfer replay from retained spot allocation, abundance and posterior means; optional paired plot |
| `workflows/bayesprism/` | Actual locally patched pybayesprism0.1.0, fixed reference summaries, preparation/run/postprocessing and synthetic tests |
| `workflows/plots/` | Numeric bulk/S1–S3, scRNA and spatial figure sources; no Word editing or publication-history scripts |
| `examples/` | Complete small inputs for direct scoring without labels |
| `provenance/` | Source/model hashes, recorded runtime and model/measurement identities |

Each workflow has its own README with exact execution commands. Refitting and raw BayesPrism execution are explicit optional commands, excluded from default replay. The compact evaluation inputs are retained because exact score/bootstrap reproduction requires them; they are not the original raw datasets.

## Scientific identity and limitations

Current scRNA is fixed18＋malignant-fraction weighted MLH1, training-only StandardScaler and shrinkage LDA. Specimens define AUROC; patients group CV/bootstrap. Original primary fold objects were not saved; separately identified q95 sensitivity refits reproduce the archived OOF probabilities and retain their own identity. Archived ecological features use an older TOSICA family supporting18/18 features; the newer seed2026 reference supports13/18 and cannot substitute missing types with zero. Raw TOSICA/CNV regeneration is not a supported pipeline in this lean repository; its provenance and full sources remain in the separate complete archive.

Bulk uses the adopted12-subtype MLP and eight-cohort LOCO MMR4 with fixed0.5/0.5 fusion. LOCO applies only to MMR4; subtype selection and upstream measurements remain fixed during classifier CV. The12-panel was adopted after comparison was viewed. Cohorts retain retrospective exposure. Bootstrap intervals condition on the stored measurements/models. All explicitly configured current seeds are2026 and resamples advance a single RNG stream.

Spatial inputs are reconstructed proxies. Replay supports computational/technical transfer, not measurement equivalence. Reference epoch-level loss history was not recovered; convergence is not asserted. Historical BayesPrism installation commit and complete dependency snapshot were not retained. `requirements.txt` records the tested current replay environment, not every historical fitting environment.

Method references: Chu et al., *Nature Cancer*2022, doi:10.1038/s43018-022-00356-3; actual Python implementation pybayesprism0.1.0, https://github.com/ziluwang829/pyBayesPrism. Third-party notices/licenses are retained with the vendor. No project-wide license has been inferred or assigned to the original project code.
