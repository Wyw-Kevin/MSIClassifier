# Weighted scRNA classifier transfer to spatial data

User request: complete spatial migration after adopting fixed18 plus malignant-
fraction weighted MLH1. Bulk DECODE is out of scope and remains unchanged.

## Locked reconstruction

- GSE178341 is the only reference. Retain the 71 non-epithelial classes in the
  existing unmerged min50 contract. Add two tumor epithelial states from the
  corrected Stromal-only q95 analysis: candidate malignant and other scored
  epithelial cells. Unscored epithelial cells are excluded, not labelled normal.
- Sample at most 1200 reference cells per class, seed 2026. Fit the reference
  NB regression with patient/sample batch, as in the existing cell2location
  pipeline. Reuse its gene filter and 180 epochs; require MLH1 to pass the filter.
- Refit each of 14 spatial sections separately, 500 epochs and 200 posterior
  draws, N_cells_per_location=8, detection_alpha=20, all explicit seeds 2026.
  These reproduce the historical training budget, not a guarantee of convergence;
  retain histories and report loss trends. Do not tune against inferred MSI groups.
- Reconstruct candidate malignant counts using the official conditional
  expected-expression formula. Compute that type's per-spot log1p(CP10K)
  MLH1 over the shared filtered genes. This is an inferred type-specific spot
  profile and is not equal to observed mean single-cell log expression.
- The section feature equals sum(reconstructed MLH1 * malignant abundance)
  divided by sum(all epithelial abundance). This implements malignant-
  abundance weighted mean times malignant/epithelial abundance fraction.
- Recompute all18 subtype/parent ratios from the new spatial abundances. Use
  the frozen shared18_weighted_mlh1_2026 scaler and LDA without spatial refitting.
  Audit out-of-training-range inputs and calibration limitations; do not rescale
  spatial features to improve apparent performance.

## Evaluation and artifacts

Join inferred MSI/MSS groups only after feature construction and prediction.
Report separate section and patient-level descriptive AUROCs, with the latter
based on mean section probability. There are 7 patients, including 2 inferred MSI.
Compare a training-fitted 18-only model and the negative weighted MLH1 feature.
These are technical-transfer diagnostics, not clinical validation or independent
test evidence. Include all14 original sections, including the named low-quality
replicate; do not remove sections according to prediction quality.

Save reference identities, signatures, checkpoint weights, posterior summaries,
spot reconstruction, sample features, source-model hash, predictions, input-scale
audit, convergence histories, vector figures and a manuscript revision audit.
Do not duplicate full spatial matrices solely for convenience. Preserve prior
models and manuscript backup. No metadata timestamps are added to the manuscript.

## Completed run and limits

All 14 sections completed with the locked budget. The frozen joint classifier
has section/patient descriptive AUROC 1.000/1.000; the training-only ecology
comparator has 0.925/1.000; negative reconstructed weighted MLH1 has 1.000/1.000.
These results do not establish an incremental value over MLH1 in this cohort.
The spatial reference was rebuilt, so comparison with the historical unweighted
branch is not a controlled change of the weighting formula alone.

MLH1 is within the training min/max range in all sections. cM06 and cTNI08 are
outside their respective training ranges in all 14 sections, cM01 in 9, and cM04
in 5. Keep this input-shift evidence alongside the descriptive performance.
The completed reference regression did not retain per-epoch loss history;
this cannot be recovered by changing the logger in source code. All 14 spatial
loss histories are saved. The fixed epoch budget does not establish convergence.

Results, figure source data, vector exports and audit are in
`最终模型/report/article/weighted_spatial_update/`. Reference/checkpoint artifacts
are in `最终模型/data/stRNA/result/weighted_mlh1_transfer_seed2026/`.
