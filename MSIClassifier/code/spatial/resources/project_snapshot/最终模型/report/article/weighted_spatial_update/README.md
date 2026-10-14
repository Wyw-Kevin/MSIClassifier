# Frozen weighted scRNA classifier transfer to spatial data

This is an exploratory technical transfer, not independent clinical validation.

## Results

| Level | Model | n | MSI | AUROC |
| --- | --- | ---: | ---: | ---: |
| Section | Weighted reconstructed transfer | 14 | 4 | 1.000000 |
| Section | 18-subtype only transfer | 14 | 4 | 0.925000 |
| Section | Negative weighted MLH1 proxy | 14 | 4 | 1.000000 |
| Patient | Weighted reconstructed transfer | 7 | 2 | 1.000000 |
| Patient | 18-subtype only transfer | 7 | 2 | 1.000000 |
| Patient | Negative weighted MLH1 proxy | 7 | 2 | 1.000000 |

## Provenance and interpretation

- Frozen classifier SHA-256: `bb3c46aa955ae272f5f862cffa4017d5712e30f5bfaf9fd60e4ac10f96bc764b`.
- Reference: 54630 sampled GSE178341 tumor cells, 12079 genes, 71 non-epithelial types plus two epithelial CNV states.
- Other scored epithelial cells are not asserted to be biologically normal. Unscored epithelial cells were excluded.
- Candidate malignant state comes from the corrected Stromal-only q95 CNV analysis; no spatial MSI labels were used for fitting.
- Each reference type was sampled at most 1200 cells; all explicit seeds are 2026. Reference regression: 180 epochs; each section: 500 epochs, 200 posterior draws, N_cells_per_location=8, detection_alpha=20.
- Reference per-epoch history was not retained in this completed fit. Reference convergence therefore cannot be established from a loss trace. All 14 section loss traces are saved.
- Section ELBO last50 versus previous50 relative change ranges from 0.3184% to 1.5452%; this is a diagnostic, not a convergence guarantee.
- The candidate-type count reconstruction was checked against the installed official cell2location compute_expected_per_cell_type API in every section.
- MLH1 is log1p(CP10K) in the reconstructed candidate-type spot profile, using the 12079 shared filtered genes. Section input = sum(MLH1 * candidate abundance) / sum(candidate + other epithelial abundance).
- A log-transformed type-specific mixed-spot reconstruction is not the observed average single-cell log expression. Spatial normalization genes and platform differ; calibration and measurement equivalence remain unestablished.
- The 18 subtype inputs are sums of inferred subtype abundances divided by the represented parent-type abundance. The reference cannot identify an unknown/omitted subtype component.
- Weighted MLH1 training range: 0 to 0.083240295; spatial range: 0.010674655 to 0.049527835; 0/14 outside training range.
- Scaler and joint classifier were never fitted on spatial data. The ecology-only comparator was fitted only on GSE178341 with the same scaler/LDA specification.
- Labels are inferred from public sample grouping, with 7 patients and only 2 MSI. Two section probabilities are averaged per patient. All 14 sections, including the named low-quality replicate, were retained.
- Historical/current comparison changes the reference, expression reconstruction and classifier together. It is not an isolated effect of the weighting formula.
- PDFs/SVGs use vector points and text; PNGs are Word fallback images. Figure color clipping at the pooled 99th percentile is display-only.

## Files

- section_predictions.csv and patient_predictions.csv: continuous scores and grouping.
- spatial_classifier_features.csv: exact 19 classifier inputs.
- feature_scale_audit.csv and convergence_audit.csv: input and optimization diagnostics.
- historical_current_comparison.csv: archived unweighted versus current weighted descriptive AUROCs.
- Figure04_weighted_spatial.pdf/.svg/.png: spatial distributions, scores and descriptive AUROCs.
- Reconstruction artifacts: `/home/wyw/rstudio/MSI/最终模型/data/stRNA/result/weighted_mlh1_transfer_seed2026`.
- Reconstruction entry point: model_code/run_weighted_spatial_transfer.py; summary: model_code/summarize_weighted_spatial_transfer.py.

Official method references: https://cell2location.readthedocs.io/en/latest/notebooks/cell2location_tutorial.html and https://cell2location.readthedocs.io/en/latest/cell2location.html.
