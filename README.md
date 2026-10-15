# MSIClassifier

Current scRNA model:17 subtypes plus weighted MLH1, developed only in GSE178341 and evaluated externally in GSE132465/GSE200997; see [model version and OOF selection limits](docs/CURRENT_MODEL.md).

MSIClassifier generates colorectal cancer MSI/dMMR research scores from bulk, single-cell and spatial gene expression. It constructs cell subtype and MMR-related features, then applies the supplied classifiers. MSI labels and BAM files are not required.

[中文说明](README_zh.md)

## Installation

Tested on Linux with Python 3.12. Run all commands from the repository directory.

```bash
conda create -n msiclassifier python=3.12 pip
conda activate msiclassifier
cd MSIClassifier
python -m pip install -r requirements.txt
```

For single-cell or spatial expression analysis, also install:

```bash
python -m pip install -r requirements-expression.txt
```

CPU is the default. Add `--device cuda:0` for a compatible GPU. [Detailed setup](docs/EXPRESSION_WORKFLOW.md).

## Reference resources

Download the required files from this repository's **Releases**:

| Analysis | Required reference archives |
| --- | --- |
| Bulk expression | Included in the code package |
| Single-cell expression | `MSIClassifier_RawReference_part1.zip` and `MSIClassifier_RawReference_part2.zip` |
| Spatial expression | `MSIClassifier_RawReference_part2.zip` |

Extract the required archives **inside the MSIClassifier directory**. The resulting path must include `resources/raw_reference/manifest.json`.

## Prepare input

| Analysis | Input |
| --- | --- |
| Bulk | Gene-by-sample CSV/TSV, with gene symbols in the first column |
| Single-cell | h5ad, 10x HDF5/MTX, loom or cell-by-gene CSV/TSV; specimen and tissue metadata |
| Spatial | One section's raw spot counts in the same supported matrix formats |

For bulk, specify `--scale linear`, `log2` or `log2p1` to match the supplied expression. Process raw microarray scanner files before use; processed probe matrices additionally need `--probe-map`.

Single-cell input needs `Sample` and `Class` fields in `adata.obs` or a `--metadata` CSV. `Class=Tumor` identifies tumor tissue. Use `--scale counts` for raw counts or `--scale log1p-cp10k` for already normalized expression. Spatial input requires raw counts.

[Input formats and examples](docs/EXPRESSION_WORKFLOW.md) · [All parameters](docs/EXPRESSION_PARAMETERS.md)

## Run prediction

```bash
# Bulk: this example assumes log2(linear expression + 1)
python predict.py bulk --expression bulk_expression.csv --scale log2p1 \
  --dataset MY_BULK --output-dir output/my_bulk

# Single-cell: read raw counts from a h5ad layer
python predict.py scrna --expression cells.h5ad --layer counts --scale counts \
  --dataset MY_SCRNA --output-dir output/my_scRNA

# Spatial: read one section's raw counts
python predict.py spatial --expression section.h5ad --layer counts \
  --sample-id sectionA --output-dir output/sectionA
```

If counts are stored in `X`, omit `--layer counts`. Use a new output directory; add `--check-only` to inspect input without running models. [Runnable bundled examples](examples/expression/README.md).

## Outputs

Each run writes `features.csv`, `scores.csv` and `run_manifest.json`. Higher scores indicate stronger model support for MSI/dMMR. Scores are research outputs, not calibrated clinical probabilities; spatial scores use reconstructed proxy features.

New bulk cohorts receive the subtype score. Optional fixed 0.5:0.5 MMR4 fusion requires matching cohort-specific parameters; [fusion usage](docs/EXPRESSION_WORKFLOW.md#optional-matching-cohort-mmr4-fusion).

## Documentation

- [Complete workflow and input files](docs/EXPRESSION_WORKFLOW.md)
- [Parameters](docs/EXPRESSION_PARAMETERS.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Already measured features and advanced usage](docs/TUTORIAL.md)
- [Reproduce paper analyses](docs/REPRODUCING_ANALYSES.md)

## License

[MIT License](LICENSE). Third-party components retain their original licenses; see [Third-party notices](THIRD_PARTY_NOTICES.md).
