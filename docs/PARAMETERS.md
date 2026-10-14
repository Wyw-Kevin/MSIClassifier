# Command and model parameter reference

[Main README](../README.md) · [Tutorial](TUTORIAL.md) · [Input files](INPUT_FILES.md)

Command-line options are taken from the supplied scripts' actual argument declarations. Model settings below come from preserved protocols/manifests and are distinct from CLI options. No command optimizes fusion weights or deployment thresholds using your prediction labels.

## 1. Direct prediction interfaces

### `workflows/scRNA/predict_scrna.py`

| Parameter | Type | Required/default | Effect |
| --- | --- | --- | --- |
| `--feature-table` | Path | Required | Headered sample-by-feature CSV |
| `--output` | Path | Required | Write identifier/score CSV; create parents |
| `--model` | Path | Default: included `resources/model/reference/shared18_weighted_mlh1_2026/model.joblib` | Select compatible artifact; recorded seed/formula are checked |
| `--weighted-mlh1-column` | String | `weighted_MLH1` | Select the explicitly verified weighted-value input column |
| `-h`, `--help` | Flag | Off | Help and exit before model loading |

The parser's internal `--model` default is `None`; the artifact resolver supplies the bundled path. Use `--weighted-mlh1-column mean_MLH1_epithelial` only with an archived table known to store the current weighted feature under that alias.

### `workflows/bulk/predict_bulk.py`

| Parameter | Type | Required/default | Effect |
| --- | --- | --- | --- |
| `--feature-table` | Path | Required | Sample identifiers and 12 parent fractions; optional four MMR gene values |
| `--output` | Path | Required | Write `Subtype` and optional MMR4/fusion scores |
| `--model` | Path | `classifier_snapshot/models/panel12_fold0.joblib` beside the script | Full-development panel 12 artifact, with recorded fold 0/seed 2026 |
| `--mmr4-lock` | Path | None | Enable matching-cohort four-gene scoring and fixed fusion |
| `-h`, `--help` | Flag | Off | Help |

Only a recorded `panel12`, `fold=0`, `seed=2026` artifact is accepted for direct bulk scoring. For fusion, all rows must belong to the lock's `held_out_cohort`. The scaler/cutoff vectors must contain four finite entries with positive SDs. Labels are not read by these predictors.

## 2. Main analysis dispatcher

### `reproduce.py`

| Parameter | Type | Default | Effect |
| --- | --- | --- | --- |
| `--branch` | Choice | `all` | Select the workflow below |
| `--output-dir` | Path | `<repository>/output` | Write branch logs, outputs and summary |
| `--plots` | Flag | Off | Add numeric plot generation; spatial branch also plots paired spots |

| Branch | Execution |
| --- | --- |
| `all` | `bulk`, `scRNA`, `sensitivity`, `spatial`, `paired`, `stability`, `bayesprism`; add `plots` with the flag |
| `bulk` | Current fixed-checkpoint sample-score and AUROC replay |
| `scRNA` | Frozen classifier, controls, patient bootstrap and rho/VIF replay |
| `sensitivity` | Verify supplied 42 threshold/CNV sensitivity models and stored scores |
| `spatial` | Saved allocation/abundance features and frozen-score replay |
| `paired` | True fusion-minus-MMR4 paired AUROC intervals |
| `stability` | Reproduce 2,000 development selection resamples |
| `bayesprism` | Reference checks and synthetic mixture, not real-cohort inference |
| `plots` | Numerical bulk/scRNA figures |

The dispatcher sets `PYTHONHASHSEED=2026` and single-thread BLAS/OMP settings in subprocesses. These are execution settings, not model tuning. Each statistical RNG retains its recorded 2026 initialization/advancing-stream behavior.

## 3. Individual stored-input workflows

| Script | Options | Default/requirement |
| --- | --- | --- |
| `workflows/bulk/reproduce_frozen_bulk.py` | `--output-dir` | Default `outputs/frozen_bulk` beside the script |
| `workflows/scRNA/replay_frozen_scRNA.py` | `--project-root`, `--output-dir` | Default resource root beside the script; output optional |
| `workflows/spatial/reproduce_frozen_spatial.py` | `--output-dir`, `--figures` | Default `outputs/frozen_replay` beside the script; figures off |
| `workflows/bulk/compute_paired_increment.py` | `--replay`, `--output` | Use both for portable replay and a fresh output directory |
| `workflows/bulk/compute_selection_stability.py` | `--output` | Default `outputs/selection_stability` beside the script |
| `workflows/bayesprism/validate_implementation.py` | `--output-dir` | Required |
| `workflows/plots/reproduce_plots.py` | `--output-dir` | Required |

The scRNA replay's `--project-root` selects an alternative resource root with the same flattened `diagnostics/`, `article/` and `model/` hierarchy. It is not the raw h 5ad project directory used by the BayesPrism environment variable.

`compute_paired_increment.py` retains old non-replay path defaults for provenance; always supply `--replay` in this repository. Its output parent must exist, so use the dispatcher or create the parent first. Direct branch defaults are not all the same directory as the main dispatcher; use explicit output options to control location.

## 4. Refitting interfaces

| Command | Parameters | Purpose |
| --- | --- | --- |
| Bulk `classifier_snapshot/run_analysis.py` | `--replay` flag, `--output PATH` | Refit preserved 7/12 comparison and historical 10 numerical control from source copies |
| Bulk `classifier_snapshot/verify_analysis.py` | Positional `ANALYSIS_DIRECTORY` | Verify generated source/models/metrics/bootstrap/workbook/figure records |
| scRNA sensitivity `run_analysis.py` | Required `--output-dir PATH` | Refit current threshold/CNV sensitivity classifiers from saved feature measurements |
| scRNA sensitivity `verify_analysis.py` | `--analysis-dir PATH`, required `--output PATH` | Verify the selected analysis directory and write a separate JSON |

The bulk refit's parser defaults to the script directory and `--replay=False`; these are not the safe output/source choices for a fresh repository run. Supply a separate `--output` plus `--replay` as shown in the tutorial. The scRNA verifier defaults to its own analysis directory for checking the bundled results.

## 5. Fixed classifier settings

| Model setting | scRNA | Bulk subtype |
| --- | --- | --- |
| Input panel | Fixed 18 subtype proportions plus weighted MLH1 | Adopted 12 parent-normalized subtype abundances |
| Scaling | Training StandardScaler | Training StandardScaler |
| Classifier | LDA `solver=lsqr`, `shrinkage=auto` | MLP one hidden layer `(4,)`, ReLU, Adam |
| CV | Existing patient-grouped five folds | Existing development five folds |
| Regularization | Automatic covariance shrinkage | `alpha=1.0` |
| Maximum iterations | Deterministic LDA fit | `max_iter=5000` |
| Learning rate | Not applicable | `learning_rate_init=0.001` |
| Stopping | Not applicable | `tol=1e-4`, `n_iter_no_change=10`; no early stopping |
| Explicit seed | 2026 where configured | `random_state=2026` |

Scalers are fitted only on training data, within training folds. Prediction does not refit them. CV conditions on fixed upstream panels/measurements. The full-development bulk model does not produce the same development scores as OOF models.

The current bulk candidate source screen is frozen scRNA oriented AUROC strictly `>0.70`, giving 17 candidates. The panel 12 rule is oriented AUROC strictly `>0.50` in both development datasets. This is directional concordance, not a significance test. Candidate selection is not a direct-scoring CLI parameter.

## 6. MMR4, fusion and bootstrap settings

| Parameter | Recorded value/meaning |
| --- | --- |
| Gene order | MLH1, MSH2, MSH6, PMS2 |
| Pre-Z expression | Original log2 matrix values |
| Cohort normalization | Mean and sample SD, `ddof=1`; target normalization does not use outcomes |
| Cutoff grid | Each gene −4 to 0, step 0.5; 6,561 four-dimensional vectors |
| Fitting unit | Omit one of 8 cohorts and fit on the remaining 7 |
| Objective | Equal-cohort training macro-AUROC; then macro-BA at score 0.5 |
| Tie handling | Round both ranking metrics to 12 decimals before comparison; choose lexicographically smallest cutoff vector |
| Temperature | Fixed 0.5 |
| MMR4 aggregation | Maximum of the four sigmoid scores |
| Fusion weights | Subtype 0.5, MMR4 0.5 |
| Pairing/bootstrap | Same sample IDs/indices for both scores; 2,000 stratified draws; percentile 95% intervals |
| Selection bootstrap | 2,000 draws within each development cohort/outcome; fixed 17 candidates/directions |
| Random stream | Seed 2026 once per recorded stream; advance without fold/replicate offsets |

```text
z_g = (log_expression_g − cohort_mean_g) / cohort_sample_SD_g
gene_score_g = 1 / (1 + exp(−(cutoff_g − z_g)/0.5))
MMR4 = max_g(gene_score_g)
Fusion05 = 0.5 × Subtype + 0.5 × MMR4
```

These settings are implemented in optimizer/locks and protocols; there are no `--temperature`, `--fusion-weight`, `--auc-cutoff` or `--bootstrap-seed` prediction flags. Changing them creates a different analysis/model.

## 7. BayesPrism environment and positional arguments

The real-cohort helper scripts are configured through environment variables and saved inputs, not argparse options. `prepare.py`, `launch.py` and `postprocess.py` are stage commands; do not use a guessed `--help` to launch them.

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `MSI_PROJECT_ROOT` | Parent-derived path; set explicitly for raw runs | Directory containing reference h 5ad and `bulk_data/<cohort>/expression.csv` |
| `MSI_BAYESPRISM_WORKDIR` | `workflows/bayesprism/runs/rebuild` | Fresh preparation/inference/cache directory |
| `MSI_BAYESPRISM_EXPORT_DIR` | `<workdir>/exports` | Postprocessed parent-relative features |
| `MSI_BAYESPRISM_COHORTS` | `current8` | Current eight cohorts or a comma-separated supported subset |

Supported current IDs are `TCGA-COADREAD`, `GSE39582`, `GSE24550`, `GSE25071`, `GSE27544`, `GSE35896`, `GSE39084`, `GSE75316`. `historical9` remains a compatibility selector adding GSE92921; it is not the paper's current eight-cohort evaluation. No arbitrary new-cohort ID or scale autodetection is supplied by this stage wrapper.

```bash
python workflows/bayesprism/run_cohort.py DATASET N_CORES
```

`DATASET` is required; `N_CORES` is an optional positive integer, default 24. The supplied launcher hardcodes 3 simultaneous cohorts with 24 workers each. Use the individual command to select fewer workers; there is no launcher `--threads` flag.

| Inference setting | Recorded value |
| --- | --- |
| Port | Locally patched pybayesprism 0.1.0 |
| Reference | 71 non-epithelial types plus Epi; raw counts |
| Reference update | `key=None`, `update_gibbs=True`; separate target-cohort updates |
| Chain parameter | 1,000; actual archived loop updates 0–998 |
| Burn-in/thinning | 500/2; 250 retained states 500,502,…,998 |
| Alpha/MAP sigma | 1/2 |
| Seed/state allocation | 2026; advancing PCG 64 stream with saved nonoverlapping sample state segments |
| Target sampling input | Whole linear matrix converted with nearest-even `np.rint` to `int64` |
| Final abundance | Updated-reference `posterior_theta_f.theta` |

## 8. Spatial reconstruction parameters and scope

Saved-input replay has only output/figure flags. Original reconstruction settings were 73 reference types, maximum 1,200 reference cells/type, 180 reference epochs, 500 epochs/section, 200 posterior draws, `N_cells_per_location=8`, `detection_alpha=20`, seed 2026 and 12,079 model genes for candidate CP10K normalization.

These values describe the preserved raw reconstruction; the compact replay does not expose them as training flags or rerun that optimization. It uses saved allocation totals, abundance/parameter means and the frozen classifier. Reference epoch history was not recovered; parameter values and finite section histories do not establish convergence.
