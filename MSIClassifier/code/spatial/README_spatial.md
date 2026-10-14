# Current weighted spatial transfer

This folder packages the completed exploratory spatial branch `weighted_mlh1_transfer_seed2026` and its actual frozen `shared18_weighted_mlh1_2026` classifier. It preserves the distinction between inferred spatial measurements and observed single-cell measurements. Packaging did not refit a model or draw a posterior sample.

## Portable verified replay

From the `MSIClassifier` folder, using Python 3.12 and the versions in `requirements_replay.txt`:

```sh
python code/spatial/reproduce_frozen_spatial.py --output-dir code/spatial/outputs/frozen_replay
```

To also recreate the retained paired spatial plot:

```sh
python code/spatial/reproduce_frozen_spatial.py --output-dir code/spatial/outputs/frozen_replay --figures
```

Add the dependencies in `requirements_figures.txt` for figures. The paired plot source uses Fontconfig's `fc-match` to resolve Arial; the original Linux environment provides it. All prediction and feature checks work without a GPU, cell2location, PyTorch or raw expression matrices. The plotting command invokes only the retained source's `spatial()` function, and does not call its bulk or manuscript publication functions.

The replay reconstructs all 18 subtype/parent ratios and weighted MLH1 from saved spot/abundance tables, verifies posterior mean abundances and the 500-epoch histories, applies the actual classifier to the exact archived 19-feature input table, and averages the two section probabilities for each patient. The archived feature table is deliberately used for probability replay, preserving the original float32/CSV provenance; independently reconstructed values validate its meaning. Outputs include section/patient predictions, a rebuilt feature table and `verification.json`.

Verified coverage: 14 sections, 7 patients, 20,733 spots. Maximum section and patient probability errors are both 2.22×10⁻¹⁶. Maximum weighted feature error from saved allocated counts is 4.70×10⁻⁹. With `--figures`, all 20,733 plotted display values and epithelial-enrichment display masks match the retained source data. These are reproduction checks of existing results, not new performance estimates.

## Scientific identity

- GSE178341 is the reference: 71 unmerged non-epithelial types plus `Epi_CNV_candidate` and `Epi_other`, 73 types total. The epithelial states derive from the approved Strom-only q95 CNV analysis; `Epi_other` is not confirmed normal epithelium.
- The reference model uses `GSE178341.h5ad` **`layers['counts']`**, copied into its input `X`; the already transformed original `X` is bypassed. There are 54,630 sampled reference cells, at most 1,200 per type.
- Reference gene pool: GSE178341 gene symbols ∩ first spatial section gene symbols ∩ `bulk_data/standardized_expression_common_genes.txt`. Duplicate symbols are summed. The reference filter uses cell_count_cutoff=15, cell_percentage_cutoff2=0.03 and nonz_mean_cutoff=1.12, leaving 12,079 genes. All 14 sections contain the same ordered model gene set; MLH1 passed the filter.
- Reference budget: 180 epochs, patient/sample batch through `reference_sample`; each section is reconstructed separately for 500 epochs, 200 posterior draws, N_cells_per_location=8 and detection_alpha=20. The active weighted entrypoint sets every explicit RNG to2026. No external MSI labels select these settings.
- Reconstruction uses the parameter **posterior means** (`post_sample_means`) and the official conditional count-allocation formula, not q05 or a mean of separately reconstructed posterior draws:

  `C_candidate[i,g] = X[i,g] × (w_mean[i,candidate] × signature[g,candidate] × m_mean[g]) / (sum_f(w_mean[i,f] × signature[g,f]) × m_mean[g] + s_mean[g])`

  `ReconstructedMLH1_i = ln(1 + 10000 × C_candidate[i,MLH1] / sum_{g in model genes} C_candidate[i,g])`

  `weighted_MLH1 = sum_i(ReconstructedMLH1_i × CandidateAbundance_i) / sum_i(CandidateAbundance_i + OtherEpithelialAbundance_i)`

- The CP10K denominator spans **all 12,079 shared filtered model genes**, not all original43,113 rows, the unfiltered gene intersection, MLH1 alone or the19 classifier features. The order is count allocation → candidate-profile CP10K → natural log1p → abundance weighting → frozen training-only Z-score → frozen lsqr/auto-shrinkage LDA.
- The historical column name `mean_MLH1_epithelial` in the actual artifact represents the current **weighted reconstructed MLH1 proxy**. The name is preserved to match the frozen model's exact input order; legacy spatial MLH1 features must not be substituted.
- Classifier SHA256: `bb3c46aa955ae272f5f862cffa4017d5712e30f5bfaf9fd60e4ac10f96bc764b`. `resources/feature_contract.json` records its exact features and fitted scaler parameters.

The proxy is a type-specific reconstructed spot profile, not an observed mean of single-cell log expression. External labels are public-grouping inferred; 7 patients include only2 inferred MSI. Spatial inputs and probabilities have not established measurement equivalence or clinical calibration. This is exploratory technical transfer.

## Sources and full reconstruction

`original_sources/` contains byte-identical copies with original relative directories. Origins, sizes and SHA256 are in `provenance_spatial.json`. The exported `run_weighted_spatial_transfer.py`, `summarize_weighted_spatial_transfer.py`, `fix_spatial_and_bulk_figure_layout.py`, `audit_saved_spatial.py` and `audit_checkpoint_history.py` contain only reversible path-routing edits. Their scientific source is unchanged.

The active reconstruction script imports the archived helper `original_sources/stRNA_result/run_cell2location_gse178341_spatial.py`. Its used functions load raw10x counts, collapse duplicate genes, read coordinates, train and export posterior summaries. The helper's original legacy CLI defaults are preserved as historical provenance; its CLI is **not** invoked by the active weighted script. Run the current weighted entrypoint for this branch.

For a full reconstruction in a separately prepared project directory:

```sh
MSI_PROJECT_ROOT=/path/to/isolated/full_project python code/spatial/run_weighted_spatial_transfer.py --workers 1 --worker-index 0
```

The supplied source retains the original caching behavior: existing `complete.json` files reuse completed results. A new computation needs a separately prepared working project rather than altering original artifacts. An external working root must contain the source h5ad and all14 raw10x directories, the fixed contract/CNV inputs/common-gene list/classifier at their recorded relative paths, and the approved protocol. `resources/project_snapshot/` supplies the compact configuration and derived inputs; large raw matrices and cell2location checkpoints are listed in the manifest rather than copied. The packaged default root is that snapshot. It is sufficient for lightweight replay, not a claim that omitted raw matrices are included.

The full reconstruction requires `requirements_reconstruction.txt` and a suitable GPU environment. The audited historical environment used torch2.6.0+cu124; install the appropriate PyTorch2.6.0 CUDA build for the hardware. Equal seeds do not guarantee identical GPU fits across versions/devices. This package did not rerun those fits. The archived summary script includes the original source-only ecology comparator refit; the lightweight replay uses its actual saved comparator and performs no fitting.

The `audit_saved_spatial.py` export can replay count allocation from external raw10x matrices plus existing posterior means without fitting. The exported audit writes to this package's `outputs/audit/` folder:

```sh
MSI_PROJECT_ROOT=/path/to/full_project python code/spatial/audit_saved_spatial.py
```

`audit_checkpoint_history.py` additionally requires the omitted original reference/section model.pt files. The original independently verified raw-count audit and checkpoint audit are retained in the snapshot, including tolerances and hashes.

## Training records and figures

The reference checkpoint's `history_` is empty; its per-epoch history is not recoverable from that checkpoint. All14 section histories contain500 finite epochs and match their original checkpoint histories. Last50 versus preceding50 mean-loss changes range0.3184%–1.5452%. Fixed budgets and loss trends do not establish convergence.

The paired figure's numerical source is `fix_spatial_and_bulk_figure_layout.py:spatial()`. It shows weighted reconstructed MLH1 on epithelial-enriched spots and local ecology scores on the other spots. The display partition and99th-percentile color clipping do not select model inputs. The final editable manuscript SVG preserves the paired figure's path geometry; later presentation edits only separate panel letters from titles. The original `unify_panel_letters.py` and `export_editable_figures.py` are archived to document that presentation chain and are not run by the lightweight replay.

## Dependencies and license provenance

Requirements files distinguish lightweight scoring, figures and complete reconstruction. `provenance_spatial.json` records installed distribution versions and upstream license metadata; `third_party_licenses/` retains the corresponding installed license notices. Scientific libraries are imported, not vendored. Original project source ownership/provenance is preserved; no new project license is invented. The retained original sources and JSON manifests intentionally preserve historical absolute-origin fields.
