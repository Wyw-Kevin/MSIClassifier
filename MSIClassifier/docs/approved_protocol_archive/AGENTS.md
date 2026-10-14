# MSI project workflow

For future scRNA/bulk preprocessing, migration, deconvolution and model evaluation,
follow `docs/superpowers/specs/locked-transcriptomic-workflow.md`. This is the
user-approved protocol and supersedes conflicting historical exploratory scripts.

- Every explicitly configured random seed/random_state must be 2026, including
  Python, NumPy, PyTorch, CUDA, R, sampling, splitting, estimators and bootstrap.
  Do not add fold, model, worker or date offsets. Repeated resamples must advance
  the same RNG stream rather than reset it inside the resampling loop.
- Preserve historical weights/results/manifests with their actual provenance.
  Changing a seed in source does not mean existing weights were retrained.
- Never use external MSI labels or external AUROC to select migration weights,
  cell definitions, features, preprocessing, DECODE settings or fusion weights.
- Use fixed GSE178341 reference definitions; no subtype merging; retain reference
  subtypes with at least 50 cells. Audit actual checkpoint coverage before
  claiming all 71 candidates are measurable.
- External scRNA cohorts are annotated independently with a frozen TOSICA
  reference. Do not jointly integrate them with the training cohort for the
  primary external evaluation.
- Bulk cohorts, including the two development cohorts, undergo separate DECODE
  target adaptation using a fixed reference and protocol. Never warm-start from
  weights jointly adapted to historical external cohorts. Label this as
  target-cohort unsupervised adaptation, not fully frozen single-sample inference.
- Negative DECODE inputs stop execution. Do not clip/shift negative expression.
- The primary classifier scaler is fitted on development training data only
  (within each training fold in CV). Dataset-wise Z-score is a separate,
  explicitly cohort-adaptive sensitivity analysis.
- Malignant epithelial MLH1 has no 50-cell sample eligibility threshold; zero
  malignant cells means missing. Whole-epithelial MLH1 is a labelled proxy only.
- The selected infercnvpy branch uses only stromal reference cells: original
  clTopLevel=Strom in GSE178341 and frozen TOSICA Strom labels in transferred
  cohorts. Run each cohort separately; missing stromal reference stops analysis.
  Do not include normal epithelial or R-inferCNV-derived Epi_Normal labels in
  this reference. Use the reference-cell q95 CNV score threshold for candidate
  malignant epithelial MLH1. This is a study-defined sensitivity rule, not an
  official infercnvpy cutoff. Historical combined-reference results remain
  historical and must not be relabelled as stromal-only results.
- Do not overwrite manuscripts or claim new performance until computations and
  provenance checks have actually completed.
- The current approved scRNA implementation is the fixed18 panel plus malignant-
  fraction weighted MLH1: mean(candidate malignant epithelial MLH1) multiplied
  by N_malignant/N_all_epithelial, with both counts from the same retained CNV
  cell set. Use the separately refitted training-only Z-score + shrinkage LDA in
  model/reference/shared18_weighted_mlh1_2026 under the final-model directory.
  Preserve older model identities. The completed spatial reconstruction branch
  is data/stRNA/result/weighted_mlh1_transfer_seed2026 under the final-model
  directory; its audited predictions/figures are report/article/weighted_spatial_update.
  It applies this frozen classifier to newly reconstructed spatial proxy inputs;
  it is not proof of measurement equivalence. Follow
  docs/superpowers/specs/weighted-spatial-transfer.md for this branch. Legacy
  spatial API/checkpoints retain their old identity; do not feed their old MLH1
  features into the new classifier. Report weighted analyses as exploratory,
  not untouched independent validation.
