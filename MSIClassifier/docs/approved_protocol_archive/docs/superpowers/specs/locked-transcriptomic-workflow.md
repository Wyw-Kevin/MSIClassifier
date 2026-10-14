# Fixed transcriptomic processing and evaluation protocol

Status: approved by the user for subsequent work. This document fixes the
workflow, not the numerical results of experiments that have not yet run.

## Reproducibility and preservation

Use 2026 for all explicit random seeds: Python random, NumPy, torch CPU/CUDA,
R set.seed, model random_state, sampling, grouped folds, permutation and bootstrap.
Set PYTHONHASHSEED before process startup when required. Configure deterministic
operations where supported and record software, hardware and exceptions; equal
seeds alone do not guarantee numerical identity across devices/versions.
Do not derive seeds by adding dates, folds, workers or model IDs. For repetitions,
initialize a generator once with 2026 and advance its state. Save sampled indices.
Archive original manifests and hashes. Legacy weights with another/unknown seed
remain legacy controls; they cannot be relabelled as trained with 2026.

## Reference and single-cell processing

1. Freeze the GSE178341 reference, patient IDs, cell/gene identities, subtype
   dictionary and parent mapping. No subtype merging. Keep reference subtypes
   with >=50 cells. Verify the expected 71 candidates and checkpoint coverage.
2. Retain raw counts and a documented expression layer. Audit gene identifiers,
   normalization denominator and transformation order against checkpoint training;
   do not silently normalize twice or replace inputs to old weights.
3. Technical integration, if justified, is restricted to reference construction
   and annotation. Do not regress away patient/MSI biological differences by
   treating them as technical batches. BBKNN graphs are not corrected counts.
4. New TOSICA training uses patient-grouped internal assessment and source-only
   checkpoint selection. The major and subtype checkpoints, gene orders, pathway
   masks and label dictionaries are frozen before external annotation.
5. Annotate GSE132465, GSE200997 and subsequent cohorts separately. Do not jointly
   integrate reference and evaluation cohorts. Record missing input genes, broad
   lineage and subtype probabilities, unknown calls and marker concordance.
6. Choose a rejection rule using reference-only held-out predictions. Until this
   calibration is complete, retain old argmax outputs as historical controls;
   an illustrative 0.5 cutoff is diagnostic only, not a newly selected rule.
7. Features are subtype counts divided by the corresponding parent-cell counts.
   Keep confidently assigned parent cells with unknown subtype in the denominator.
   Track uncertain parent assignments separately. A zero parent denominator and
   an unsupported subtype are missing/unmeasured, not biological zero. Record
   coverage and predefine training-only imputation/abstention before scoring.
8. MLH1 uses normalized observed expression from malignant epithelial cells.
   Epithelial identity and CNV-supported malignancy are distinct operations.
   Audit normal reference purity before promoting infercnvpy sensitivity results.
   No >=50 malignant-cell rule per sample: >=1 permits estimation, with counts
   and uncertainty recorded; zero means missing. Whole-epithelial proxy results
   must remain separately labelled. Keep CNV thresholds independent of MSI AUROC.

## Current scRNA implementation update

The approved fixed18 scRNA implementation uses weighted_MLH1 = mean MLH1 in
candidate malignant epithelial cells * N_malignant / N_all_epithelial. Both
counts must refer to the same retained CNV-analysis cell set; do not combine
sampled malignant counts with full-matrix epithelial counts. Preserve measured
zeros; absent malignant cells are missing. Keep all 18 subtype features and fit
Z-score and lsqr auto-shrinkage LDA only on the training cohort or training fold.
The active separately saved artifact is
`最终模型/model/reference/shared18_weighted_mlh1_2026/model.joblib`.
Fixed-panel patient-grouped CV does not isolate historical feature selection or
cohort-level CNV. Both external cohorts have been viewed during exploration.
Older nested, clinical-adjustment and spatial-transfer results retain their
original identities and do not become results of this weighted classifier.
The associated current results and audit are in
`最终模型/report/article/weighted_scRNA_update/`.

## Current spatial implementation

The completed experimental weighted transfer is specified in
`docs/superpowers/specs/weighted-spatial-transfer.md`. Its 73-type cell2location
reference comprises 71 non-epithelial types plus candidate malignant and other
scored tumor epithelial cells. Each section is separately reconstructed, then
the frozen current scRNA scaler/LDA scores the 18 subtype inputs and reconstructed
weighted MLH1 proxy. Do not reuse historical spatial feature tables as if they
had this new definition. Do not call other scored epithelium confirmed normal.

Audited outputs are `最终模型/report/article/weighted_spatial_update/` and raw
reconstruction artifacts are
`最终模型/data/stRNA/result/weighted_mlh1_transfer_seed2026/`.
All 14 sections are complete. Both section and patient descriptive joint AUROC
are 1.000, but weighted MLH1 alone also achieves 1.000. Seven patients with two
grouping-inferred MSI labels cannot establish clinical validity or incremental
ecological value. Some subtype inputs lie outside the single-cell training
range. Reference epoch history was not retained; all section histories were
saved. Do not claim convergence or cross-platform measurement equivalence.
Historical spatial checkpoints and legacy APIs remain separate implementations.

## Bulk deconvolution

1. Process TCGA-COADREAD, GSE39582 and each external cohort independently using
   the same frozen reference, pseudo-bulk resource and documented settings.
2. Audit platform/units, gene coverage, missing genes and transformations. Stop
   on negative DECODE inputs; no clipping or arbitrary shift. Avoid repeated
   row-max/log transformations and unverified zero padding.
3. Run Stage2 target adaptation and Stage3 separately for each target cohort.
   Initialize independently from the agreed starting configuration, not historical
   all-cohort weights. Fix hyperparameters/early stopping without target MSI labels.
4. Preserve each cohort's checkpoint and outputs. Adding a cohort must not change
   previous cohort outputs. Cohort adaptation is transductive preprocessing;
   frozen inference is a separately reported sensitivity analysis.
5. Estimate reference-defined subtype/parent features consistently. Shared labels
   do not prove identical quantitative calibration across platforms. Do not call
   independent DECODE adaptation complete cross-platform batch removal.

## Scaling, model selection and evidence

Primary scaling is training-fitted Z-score, separate for scRNA and bulk and fitted
inside training folds. Dataset-wise Z-score remains a labelled cohort-adaptive
sensitivity analysis. Do not automatically apply ComBat to proportions.

First compare preprocessing with the historical fixed feature panels/configurations
to isolate its effect. Then perform reference/development-only feature selection
and nested model/fusion selection. Final feature number is not forced to 18.
Do not filter features using GSE132465 direction or AUROC. Historical shared18
performance is retrospective and must not be relabelled prospectively selected.

For scRNA, group all cells/samples from each patient together. A fully isolated
end-to-end CV claim additionally requires reference-dependent supervised steps
to respect outer folds; otherwise call it downstream classifier CV conditional
on a fixed reference. For bulk, fully isolated CV also restricts DECODE target
adaptation to the permitted fold; separate transductive evaluations explicitly.

Preserve the existing dataset-level seed-2026 bulk split:
- Classifier fitting: TCGA-COADREAD and GSE39582.
- Model selection: GSE18088, GSE24550, GSE39084, GSE75316.
- Round-specific holdout: GSE25071, GSE27544, GSE35896, GSE92921.

All previously viewed cohorts retain historical-exposure status; re-splitting
cannot make them never-used independent evidence. Report per-cohort AUROC and
patient-level intervals plus aggregate results. Cohort labels, subtype directions,
and reciprocal TCGA/GSE39582 holdout performance assess residual platform effects;
UMAP mixing alone is not validation. Maintain paired ecology-only, MLH1/MMR4-only
and fusion ablations. Clinical sensitivity analyses remain distinct from technical
batch correction. Do not remove clinical biology from proportions by default.

## Implementation order

Reference/transfer audit -> separate two-cohort DECODE pilot -> fixed-panel
preprocessing comparison -> lock preprocessing -> rebuild classifiers -> final
evaluation -> update manuscript and figures with matching model identities.

Audit-only work does not authorize claiming retraining was completed. Preserve
historical artifacts; write new results to new directories with manifests.

## Additional 72-category bulk composition branch (2026-10-03)

User requested a separate all-cell reconstruction with one Epi category at both
subtype and major levels, retaining all71 existing non-epithelial types. All new
seeds are2026. Ten independently adapted cohort runs covering1323 samples are
complete under `最终模型/data/bulk/result/decode72_epi_seed2026`; audited exports
are in `最终模型/report/article/bulk_decode72_epi_seed2026`.

This experimental branch uses8644 commonly measured finite genes and matching
log2/max-scaled pseudobulk/bulk inputs. It is not the historical71-class measurement
and does not replace the classifier input. Major fractions are summed from the
same72 predictions; Epi is identical in subtype/mid/major whole-tissue outputs.
See `docs/superpowers/specs/2026-10-03-bulk-decode72-epi.md` for full settings.

Structural and saved-model replay checks passed, but real bulk shows marked
source-target expression shift and several cohorts have narrow, high epithelial
estimates. Biological measurement validity is NOT established. Keep these as
flagged exploratory composition estimates, not tumor purity or validated cell
fractions; do not silently feed them into the old classifier or update manuscript
performance using the old model identity.

## Reference preprocessing correction (2026-10-03)

The user subsequently authorized correction against the DECODE paper. The new
reference/pseudobulk identity is `decode_reference_paper2026`; see
`docs/superpowers/specs/2026-10-03-decode-reference-preprocessing-correction.md`.
The previous branch used log2(sum raw counts+1), which differs from the paper's
per-cell total normalization/log followed by aggregation. Its description as
matching log2 inputs above refers only to nominal transformations, not verified
measurement equivalence. Preserve that historical implementation and its results.
The new branch uses full-gene reference QC, per-cell total10000 and log1p before
feature subsetting and mixture generation. It retains explicitly local cell
labels/mixture priors; it is not a claim of full paper replication. Check its
manifests for actual completion. No corrected target models or predictions are
implied by reference preparation; old checkpoints cannot accept the new inputs.

## Audited bulk readaptation (2026-10-03)

Following user approval, all10 cohorts/1323 samples have independently completed
fresh DECODE adaptation under `最终模型/data/bulk/result/decode70_bulk_audited2026`.
The reference is the corrected70-class/8606-gene reference above. GEO processing
metadata, GSE25071 sample VALUE and official TCGA HiSeqV2 units were checked;
processed RMA/VSN/ABI/TCGA expression is retained, MAS5 GSE92921 receives one log,
then all targets receive row-max scaling. This is audited platform-specific
processing, not proof of reference/bulk measurement equivalence or full paper
replication. See `2026-10-03-bulk-audit-and-readaptation.md` in the same specs folder.

Primary outputs use the denoised unknown-cell pathway and are relative among
supported reference types. The same-checkpoint standard pathway is saved as a
sensitivity analysis. Both pathways pass saved-weight CPU replay. Exports are
`最终模型/report/article/bulk_decode70_audited2026`. Real-bulk path sensitivity
remains large; source tests show limited non-epithelial subtype discrimination
beyond a fixed-composition baseline conditional on predicted Epi. Do NOT promote
these diagnostic outputs to the old11-subtype classifier or manuscript as validated
cell fractions. No classifier performance or manuscript was updated.

## Python BayesPrism exploratory rerun (2026-10-03, in progress)

User authorized rerunning nine cohorts with GSE18088 excluded because its VSN
input cannot be inverted as ordinary log2. Branch:
`最终模型/data/bulk/result/pybayesprism9_seed2026`.
Protocol: `2026-10-03-python-bayesprism9.md` in this specs directory.
1270 samples prepared; 71 fixed non-Epi labels + Epi, original raw-count
reference, tutorial gene filters. All Epi is not malignant; key=None is a generic
mixture branch. Microarray source transformations are exploratory, not RNA-seq
counts. Preserve continuous linear expression and explicit sampling rounding.
Python port0.1.0 required documented local correctness fixes; original source is
archived, tests/seed replay passed. Formal runs have started; consult progress.json
and per-cohort complete.json before claiming completion. Exports under
`最终模型/report/article/pybayesprism9_seed2026` update only for completed cohorts.
This branch does not replace DECODE or the existing classifier/manuscript.

## User-selected bulk twelve-subtype update (2026-10-13)

The user selected the twelve-subtype bulk panel after the seven/twelve comparison.
The source candidate pool is17 unmerged GSE178341 subtypes with orientedAUROC
strictly>0.70. Freeze their source directions and retain subtypes with oriented
AUROC strictly>0.50 in both GSE39582 andTCGA-COADREAD. This is directional
concordance; it does not establish association significance.

The actual seed2026 four-unit ReLU/alpha1/Adam MLP fits and training-only scalers
are retained in
`最终模型/report/article/bulk_7_vs12_mlp_mmr4_loco_seed2026_20261013/models/`.
The selected full-development subtype checkpoint is `panel12_fold0.joblib`;
fold1–5 are the archived-fold refits. Do not substitute historical ten-feature
weights or reinterpret the exploratory DECODE estimates as these BayesPrism
subtype/parent inputs. The selected twelve-feature order is recorded in
`panels_locked_before_fit.json` and in each actual checkpoint.

Continue to use the archived eight-cohort LOCO MMR4 scores and fixed0.5:0.5
fusion. These cutoff locks represent cohort-specific evaluation rounds, not a
unified deployable new-sample MMR4 parameter set. The selected subtype panel and
measurement inputs remain outside classifierCV; this is not end-to-end LOCO.
The12 configuration was selected after comparative results were viewed and
remains exploratory; all six GEO cohorts retain retrospective exposure status.

The current manuscript/supplement and independently checked2000 development
selection resamples are in
`最终模型/report/article/bulk12_loco_manuscript_seed2026_20261013/`.
Source candidate discovery, historical cohort-adaptive measurements and fixed
source direction are conditional inputs to the stability analysis. Preserve
older ten-feature/eleven-classifier outputs under their actual historical identity.
The approved fixed18 weighted-MLH1 scRNA classifier and its spatial transfer
remain unchanged by this bulk panel adoption.
