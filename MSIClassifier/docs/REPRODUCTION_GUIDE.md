# Execution guide

## 1. Archived replay (recommended starting point)

From the package root, `python reproduce.py --branch all` runs all CPU checks and figure regeneration. Numerical comparisons stop on sample/feature misalignment or an unexpected score. Outputs and log files are written separately; each branch emits a machine-readable verification record. The runner uses its own folder for source paths, so it can be moved to another location.

| Branch | Direct command from package root | What is recomputed |
| --- | --- | --- |
| Bulk | `python code/bulk/reproduce_frozen_bulk.py --output-dir output/bulk_replay` | Actual six panel12 checkpoints; four-gene log-input scores with archived LOCO locks; fixed fusion;1212 specimen scores |
| scRNA | `python code/scRNA/replay_frozen_scRNA.py --output-dir output/scRNA_replay` | Frozen full model; separately identified q95 fold refits; archived S7 AUROCs; patient-cluster bootstrap and rho/VIF |
| Spatial | `python code/spatial/reproduce_frozen_spatial.py --output-dir output/spatial_replay --figures` |14 sections/7 patients, weighted features and frozen probabilities; optional paired spatial plot |
| Increment | `python code/bulk/compute_paired_increment.py --replay --output output/paired_increment` | Fusion-minus-MMR4 paired AUROC intervals using the exact archived2000 draws |
| Stability | `python code/bulk/compute_selection_stability.py --output output/bulk_selection_stability` | Both-development directional selection for all17 candidates, exact archived2000 resamples |
| BayesPrism | `python code/bayesprism/validate_portable.py` | Actual port import, patched implementation, counts reference, final-to-parent fractions and synthetic mixture checks |
| Figures | `python code/reporting/reproduce_publication_figures.py --output-dir output/figures` | Bulk/S1–S2 plots from saved numeric inputs, S3 from sensitivity tables, final SVG exports; source identities recorded |

`code/bulk/test_archived_score_replay.py` independently checks the specimen score oracle and published cohort AUROCs. `code/bulk/mmr4/test_loco.py` verifies threshold fitting on training cohorts, deterministic ties and fixed fusion.

## 2. Refit from stored features

This is separate from default replay and does not rebuild raw measurement:

```sh
python code/bulk/classifier_snapshot/run_analysis.py --replay --output output/bulk_feature_refit
```

It redoes the archived7/12 comparison and historical10 control on stored inputs using the recorded2026 configuration. Its outcome remains conditional on reference directions, full-development feature screening and prior deconvolution; it does not turn the cohorts into untouched external validation. Environment-specific floating-point differences may affect refitted objects; retain the supplied checkpoints for exact manuscript score replay.

Current uncapped scRNA subtype/CNV sensitivity implementation and its inputs are in `code/scRNA/project/最终模型/report/article/scRNA_AUROC_threshold_all_candidates_seed2026_20261013/`. It includes all passing39/24/17/9 subtype sets plus the fixed18 reference and q90/q95/q99 comparisons. The earlier capped implementation is in `code/scRNA_sensitivity/` for historical provenance and must not replace current results.

## 3. Raw-data branches

Read the relevant modality README before starting. Supply data and large checkpoints from `DATA_REQUIREMENTS.md`. The scRNA exported wrappers use `MSI_PROJECT_ROOT` or the documented root argument. Spatial raw reconstruction uses `MSI_PROJECT_ROOT` pointing to an isolated working project. BayesPrism uses `MSI_PROJECT_ROOT`, `MSI_BAYESPRISM_WORKDIR` and `MSI_BAYESPRISM_COHORTS=current8`; run `prepare.py`, `launch.py`, then `postprocess.py`. These are cohort adaptation/reconstruction runs, potentially requiring GPU or many CPU hours.

Historical source copies preserve their original path/default settings and are not all intended as direct launch commands. Active exported wrappers apply the recorded current protocol. Never rerun old raw pipelines by changing a seed and relabel their existing checkpoints as retrained. For spatial raw reconstruction, prepare an isolated project root with the paths listed in its README; `MSI_PROJECT_ROOT` controls that root.

The older TOSICA reference supporting all18 ecology types must be restored for faithful archived measurement reconstruction. New seed2026 checkpoints support only13 of those types; the exported annotation wrapper stops for missing required types. Raw end-to-end regeneration of the manuscript's entire fixed18 ecology is not certified by this package.

## 4. Reports and provenance

`root_source_manifest.json` and the three modality manifests identify byte-preserved sources and path-only adapters. `file_manifest.json` identifies the delivered files. `document_split_audit.json` identifies the source manuscript, separate supplementary Word and the new manuscript copy. CSV and NPZ attachments retain full precision; the Excel display is rounded. Clinical meanings and limits are in the supplementary methods; raw metadata/version gaps are not filled by inference.
