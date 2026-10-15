# Expression-input examples

Run from the repository root. Each command needs a new output directory.

These are **connection checks**, not validation cohorts or estimates of clinical accuracy. Do not use their scores to select features, cutoffs or fusion weights.

## Bulk synthetic mixture

`bulk_linear_synthetic.csv` has 17,039 gene rows and one sample. It is a synthetic mixture of two frozen reference profiles, normalized to a total of approximately30,000 rounded expression units.

```bash
python predict.py bulk --expression examples/expression/bulk_linear_synthetic.csv \
  --scale linear --dataset synthetic --workers 2 --output-dir output/expression_bulk_example
```

The tested output is `bulk_example_scores.csv`. It checks expression reading, independent BayesPrism reference update, within-parent fractions and the unchanged panel12 classifier.

## Small public single-cell subset

`scRNA_200cells_log1p_cp10k.h5ad` is a 200-cell subset of public GSE132465 specimen SMC01-T, sampled with2026 and preserving original gene-expression X and Sample/Class metadata. The selection retains broad lineages for a small pipeline connection test; it is not the full specimen. This X is already normalized natural-log CP10k expression, so do not normalize it again.

```bash
python predict.py scrna --expression examples/expression/scRNA_200cells_log1p_cp10k.h5ad \
  --scale log1p-cp10k --dataset integration_check --batch-size 64 \
  --output-dir output/expression_scRNA_example --device cuda:0
```

The saved CUDA annotation/CNV measurements have been rescored with the current17 classifier; the output is `scrna_example_scores.csv`. CPU execution is available by changing `--device`; hardware/version differences can affect CNV clustering. Install both raw-reference archives before running. This 200-cell subset score does not replace the archived full-specimen score.

## Synthetic spatial counts

`spatial_32spots_counts.h5ad` contains32 synthetic spots and12,079 genes drawn with2026 from the frozen73-type signature. It contains raw integer counts and synthetic coordinates.

```bash
python predict.py spatial --expression examples/expression/spatial_32spots_counts.h5ad \
  --sample-id synthetic_section --output-dir output/expression_spatial_example --device cuda:0
```

This actually fits500 epochs and samples200 posterior draws before reconstructing the proxy. Its run is larger than preflight or feature scoring. Use `--check-only` to inspect only the input. The full current reference and expression dependencies are required. The current17 score reuses measured features from that completed spatial run; spatial fitting was not repeated for this panel update. Numerical example scores document one tested run and are not calibration targets.
