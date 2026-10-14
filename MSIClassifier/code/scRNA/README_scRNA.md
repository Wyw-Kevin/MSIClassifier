# scRNA code and frozen evidence package

This package preserves the identifiable source used by the current paper's single-cell branch: fixed18 subtype features plus candidate-malignant-fraction weighted MLH1, training-only StandardScaler and lsqr/auto-shrinkage LDA, patient-grouped evaluation, separate stromal-only CNV measurements, controls and post hoc patient bootstrap/VIF. Packaging did not train models, choose features, alter external labels, rerun CNV, or edit a manuscript.

`originals/` contains byte-identical source copies. `project/` mirrors the original project layout and contains portable source exports plus necessary small resources. Source and packaged hashes and transitive local imports are in `provenance_scRNA.json`. Portable exports add project-root lookup and replace paths; scientific equations, model settings and seeds remain those of the source. Original absolute paths inside copied JSON manifests retain actual historical provenance.

## Fully standalone archived-feature replay

Use Python3.12 and the observed versions in `requirements_replay.txt`. From the MSIClassifier directory:

```sh
python code/scRNA/replay_frozen_scRNA.py --output-dir output/scRNA_replay
```

The command uses bundled resources by default, performs no fitting, and verifies:

- Frozen full-development classifier predictions on64 development apparent,23 GSE132465 and15 GSE200997 observations.
- Archived primary64-sample grouped OOF AUROC0.9783251232 and21 S7 control AUROC rows.
- The actual4000 saved patient-cluster bootstrap draws, paired deltas and centered p-values0.163918/0.217391; GSE200997 has no p-test because its paired delta is degenerate.
- All18 saved correlation/VIF rows; these calculations do not use outcome labels.
- Separately identified archived CNV-q95 sensitivity fold models reproduce primary OOF scores within numerical precision. **Original primary five fold-model objects were not saved in the weighted branch.** The sensitivity refits retain their own model identity and are not relabelled as the original primary fits.

The main AUROC point estimator uses64 specimen scores, not62 patient means. Patients define CV groups and bootstrap clusters, with repeated specimens travelling together. Main confidence intervals and S7 confidence intervals arise from separate archived patient-bootstrap loops; their slightly different bounds are retained.

The replay was verified both in this package and after an actual copy to another directory with `/tmp` as the working directory. Frozen prediction differences were at most1.11e-16; separately identified q95 fold-refit differences were at most2.89e-15. See `verification/`.

## Score an already measured complete feature table

```sh
python code/scRNA/score_weighted_features.py --feature-table features.csv --weighted-mlh1-column weighted_MLH1 --output predictions.csv
```

The table must contain unique `sampleID`, all18 subtype columns and the explicitly weighted MLH1 column. The weighting is `mu*N_candidate/N_all_epithelial`, with counts from the same retained CNV cell set. Missing/unmeasured columns or non-finite values stop scoring; no zero filling, imputation, proxy substitution, cohort re-scaling or refitting is performed. The artifact's historical slot name `mean_MLH1_epithelial` contains **weighted** MLH1 in current archived weighted-feature files. To replay such a file, explicitly pass `--weighted-mlh1-column mean_MLH1_epithelial`. Do not use the same option with old raw-MLH1/proxy files.

## Two actual frozen TOSICA reference families

The families are distinct and must not be mixed.

| Purpose and actual source | Subtype model classes | Genes | Current fixed18 dictionary coverage |
|---|---|---|---|
| Older `scRNA_model/reference` family, used to construct archived GSE132465 and GSE200997 ecology features | TNKILC26, Myeloid10, Strom33, Plasma3 |3000 per subtype model;16496 major model |18/18 |
| `reference_seed2026` family, used for current transferred Strom labels in CNV and the2026 migration sensitivity | TNKILC9, Myeloid6, Strom5, Plasma2, including Other labels |16496 |13/18 |

The2026 family lacks cTNI14,cTNI18,cTNI16,cTNI08,cTNI19. Fresh2026 annotation cannot reconstruct all current18 subtype inputs. Its coverage must not replace the older archived ecology measurements or be silently encoded as biological zero. `reference_coverage.json` records actual dictionaries. Coverage of18 classifier features does not establish measurement of all71 candidates.

Lineage evidence:

- `run_unmerged_min50_scrna_pipeline.py:263–275` reads the original `result/process/GSE132465_tosica_migration_cells.csv`; its migration manifest identifies the older frozen reference family.
- `diagnostics/GSE200997_locked_shared18/process/GSE200997_migration_manifest.json` identifies that same older family, and its `frozen_protocol.json` preserves actual historical checkpoint hashes.
- `run_infercnvpy_malignant_reference.py:67–82` separately reads2026 migration Strom labels for the CNV reference. GSE178341 uses its original Strom annotation.

Thus the current frozen classifier replay uses preserved older ecology feature tables together with the later stromal-only CNV-weighted MLH1 measurement. Historical checkpoint training seeds and selection exposure retain their actual recorded/unknown provenance; they are not relabelled as newly trained2026 models. Some historical source modules are retained solely to close imports or document earlier selection.

## Independent annotation of raw cohorts

A lightweight inventory check needs no GPU or raw h5ad:

```sh
python code/scRNA/annotate_frozen.py --reference-family historical_features --check-only
python code/scRNA/annotate_frozen.py --reference-family seed2026_cnv --check-only
```

The checks report omitted checkpoint files and coverage. Actual cell annotation requires the exact omitted weights:

```sh
python code/scRNA/annotate_frozen.py --project-root /path/to/full/project --reference-family historical_features --dataset GSE132465 --h5ad /path/to/GSE132465.h5ad --sample-col Sample --tumor-col Class --output-dir output/GSE132465_historical_annotation
python code/scRNA/annotate_frozen.py --project-root /path/to/full/project --reference-family seed2026_cnv --dataset GSE200997 --h5ad /path/to/GSE200997.h5ad --sample-col Sample --tumor-col Class --output-dir output/GSE200997_seed2026_annotation
```

Each invocation annotates its own cohort independently with frozen weights. The wrapper calls the original `scRNA_msi.preprocess` gene-order alignment/model inference functions, uses the documented>=200 sample-total-cell QC, and writes cell labels/probabilities only. It does not produce a legacy whole-epithelial MLH1 proxy or score a classifier. `--require-fixed18-coverage` stops when a chosen reference lacks panel features. New explicit RNG initialization uses2026 without offsets.

Required large assets are inventoried, not bundled:

- GSE178341.h5ad, GSE132465.h5ad and GSE200997.h5ad with the original X, counts and cell/gene metadata. The h5ad matrices are not redistributed here.
- Historical frozen major model_best.pth and subtype model-9.pth/model-14.pth files; these are the actual older family used for archived ecology features.
- Five separate2026 model checkpoints, approximately507MB each. Their subtype model-9/model-14 filenames are aliases to best checkpoints, not evidence that those historical epoch numbers were selected.
- Training-only common-gene reference h5ad, if inspecting/reconstructing the original TOSICA training procedure.

Both families' gene orders, labels, pathway masks and recorded model configs are bundled. Full third-party TOSICA source, pathway GMT resources and MIT license are retained under `project/最终模型/scRNA_model/reference/TOSICA_source`. Relative `project/TOSICA` and2026 reference directory aliases point within the bundle. Preserve these aliases when transporting the full raw-processing tree; their targets contain bundled metadata. Large weights must be supplied at the inventory's exact matching locations or via a full `--project-root`.

Observed optional-stage dependencies are in `requirements_raw_processing_observed.txt`; this file does not invent missing historical fit-time software versions. Raw annotation benefits from CUDA; CPU inference is supported by the inherited device selection, but no raw/GPU inference was rerun during packaging.

## CNV and weighted-expression source

The portable sources live under `project/最终模型/model_code/`:

- `run_infercnvpy_malignant_reference.py`: separate-cohort stromal reference, sample-total>=200 QC, initial100000-cell tumor-tissue draw and reference union/deduplication. The initial draw includes all lineages, not malignant-only cells. GSE178341's actual union contains111727 unique cells,105898 from tumor specimens.
- `recompute_mlh1_from_stromal_infercnvpy.py`: recompute reference q95 and candidate epithelial MLH1 from original X; no>=50 malignant-cell eligibility threshold. Existing q99 output columns remain historical; downstream q95 is explicit.
- `evaluate_inverse_weighted_mlh1.py --weighting malignant_fraction`: source of the separately refitted current19-input implementation. It is included for provenance; **do not run it as part of frozen replay**, as it performs new raw and weighted fits.
- `prepare_weighted_scrna_article_results.py`: matched four-model ablations and patient-cluster intervals; this source also fits ablation classifiers and is not needed for frozen-score replay.

For an explicitly requested raw CNV rerun, use a separate working project root containing the matching raw inputs and archived feature resources. The source honors `MSI_PROJECT_ROOT` and its original stage-specific environment options:

```sh
MSI_PROJECT_ROOT=/path/to/copied/full/project INFERCNVPY_REFERENCE_MODE=stromal_only INFERCNVPY_OUTPUT_DIR=/path/to/new/CNV_results python code/scRNA/project/最终模型/model_code/run_infercnvpy_malignant_reference.py
MSI_PROJECT_ROOT=/path/to/copied/full/project INFERCNVPY_BRANCH_DIR=/path/to/new/CNV_results INFERCNVPY_MLH1_REFERENCE_QUANTILE=0.95 INFERCNVPY_MLH1_DATASETS=GSE178341 INFERCNVPY_MLH1_OUTPUT_LABEL=training_unique python code/scRNA/project/最终模型/model_code/recompute_mlh1_from_stromal_infercnvpy.py
MSI_PROJECT_ROOT=/path/to/copied/full/project INFERCNVPY_BRANCH_DIR=/path/to/new/CNV_results INFERCNVPY_MLH1_REFERENCE_QUANTILE=0.95 INFERCNVPY_MLH1_DATASETS=GSE132465,GSE200997 INFERCNVPY_MLH1_OUTPUT_LABEL=external_unique python code/scRNA/project/最终模型/model_code/recompute_mlh1_from_stromal_infercnvpy.py
```

The gene-coordinate file `inferCNV/GSE132465/geneFile.txt` is bundled. Missing Strom reference stops the inherited CNV stage. Raw fresh end-to-end primary reconstruction was not verified: large weights/raw data are omitted, historical reference identities differ from2026 annotations, and historical upstream fit/normalization records are incomplete. No measurement-equivalence or end-to-end isolated validation claim follows from this packaging.

## Controls, figures and threshold sensitivities

- S1–S9 original audit/control source: `project/最终模型/report/article/revision_review_20260928/audit_evidence.py`; its source needs raw h5ad for layer/count audits and refits three extra controls. Archived outputs and predictions are supplied for no-refit replay.
- Centered paired patient-bootstrap p-values and label-free VIF/correlations: `project/最终模型/report/article/第三版_中文版意见修订/prepare_evidence.py`. It uses fixed saved scores/counts, a single advancing2026 RNG stream and saves patient draw indices.
- Descriptive four-gene expression panels: `project/最终模型/report/article/mmr4_unweighted_boxplots_20260929/plot.py`, reading the same saved candidate-cell means. Original figure assembly/Word insertion code is preserved under `originals/` only; replay does not edit Word files.
- The real related `compare_mmr4_loss_mlh1_cutoff.py` analysis predicts **stored MLH1 loss/express22/42**, not MSI/dMMR35/29. Its archived inputs/model/results retain this distinct target. It does not establish the increment of adding MSH2/MSH6/PMS2 to the current weighted classifier.
- Current uncapped subtype-threshold and CNV sensitivity source, inputs,42 saved models and verification code are under `project/最终模型/report/article/scRNA_AUROC_threshold_all_candidates_seed2026_20261013/`. This is the39/24/17/9 subtype analysis. Its `run_analysis.py` performs optional new fits; frozen replay only reads the saved models. The earlier capped branch is not copied here and is coordinated separately by the package maintainer.

Limits retained: original primary OOF fold objects missing; original TOSICA training has older defaults overridden by its2026 wrapper and is not a newly endorsed AGENTS-compliant training protocol; some model/reference selection was historically exploratory; CNV Leiden resolution was not explicitly saved; normalization creation logs and dated QC-rule decision records were not recovered; cohort-level CNV is fixed in CV. Replays are conditional evidence checks, not new performance estimates or untouched external validation.
