# Expression-input command parameters

[Workflow and file schemas](EXPRESSION_WORKFLOW.md) · [Stored-input parameters](PARAMETERS.md)

## Shared parameters

Run `python predict.py {bulk,scrna,spatial} --help` for the corresponding complete parser.

| Parameter | Required/default | Meaning |
| --- | --- | --- |
| `--expression PATH` | Required | Gene/probe-by-sample text for bulk; cell/spot matrix for scRNA/spatial |
| `--output-dir PATH` | Required | New or empty run directory |
| `--check-only` | Off | Validate input and stop before model/reference execution |
| `--resource-dir PATH` | `resources/raw_reference` | Extracted frozen raw-reference archive; bulk uses its already bundled reference |

## Bulk

| Parameter | Required/default | Meaning |
| --- | --- | --- |
| `--dataset ID` | Required | Dataset identifier carried to outputs; a fusion lock must match exactly |
| `--scale {linear,log2,log2p1}` | Required | Explicit input units/transform; no scale auto-detection |
| `--probe-map PATH` | None | `probe_id,gene_symbol` CSV; one gene per probe, median per gene in supplied scale |
| `--workers N` | 4 | Positive BayesPrism multiprocessing worker count |
| `--mmr4-lock PATH` | None | Optional archived matching-cohort scaler/threshold lock |

Hard-coded scientific settings remain reference72, `key=None`, independent target update, 1000/500/2 chain/burn/thinning, alpha1, sigma2 and seed2026; features12 and the complete-development classifier are unchanged. A minimum1000 shared reference genes is a computational gate rather than evidence of biological validity.

## Single-cell and spatial file readers

| Parameter | Required/default | Meaning |
| --- | --- | --- |
| `--format` | `auto` | `auto,h5ad,10x-h5,10x-mtx,loom,text`; .h5 means a 10x HDF5 structure |
| `--layer NAME` | None, use X | Explicit AnnData layer to read |
| `--transpose` | Off | Transpose text tables from gene-by-cell/spot to cell/spot-by-gene |
| `--gene-symbol-column NAME` | None | Use this var field as unique gene symbols |
| `--device {cpu,cuda:0}` | `cpu` | Actual execution device; unavailable CUDA raises an error |

## Single-cell only

| Parameter | Required/default | Meaning |
| --- | --- | --- |
| `--dataset ID` | Required | Source/new dataset ID; GSE178341 uses its original annotations |
| `--scale {counts,log1p-cp10k}` | Required | Raw counts normalized once over the full library, or already normalized natural-log CP10k |
| `--metadata PATH` | None | Optional cell-indexed CSV providing sample/tissue fields |
| `--sample-column NAME` | `Sample` | Specimen identifier in obs/metadata |
| `--tissue-column NAME` | `Class` | Tissue origin, not MSI outcome |
| `--tumor-value VALUE` | `Tumor` | Exact tumor designation |
| `--batch-size N` | 1024 | Positive TOSICA inference batch size |

Fixed values: specimen QC >=200 total cells; historical frozen ecology annotation; separately frozen2026 Major for external Strom/Epi CNV calls; up to100000 tumor-tissue cells and20000 reference cells; Strom-only infercnvpy with lfc_clip3, window100, step10, dynamic threshold1.5, excludechrX/Y, jobs4; PCA/neighbors/Leiden seed2026; reference q95 boundary tolerance1e-12. All18 features and the current weighted MLH1 definition remain fixed. Complete features use the unchanged training-only scaler and LDA. Samples with missing required inputs abstain.

## Spatial only

| Parameter | Required/default | Meaning |
| --- | --- | --- |
| `--sample-id ID` | Required | Section identifier; one section per command |
| `--positions PATH` | None | Optional spot-indexed `x,y` CSV |

Only raw counts are accepted. Fixed values: saved73-type reference signature; at least2000 shared signature genes includingMLH1; new section Cell2location500epochs, batchmin(1024,n_spots), lr.002, train_size1, 200posterior samples, N_cells8, detection_alpha20, seed2026. The reference regression and current classifier are not refitted. Posterior-mean conditional allocation is verified against the official implementation; all shared retained genes define CP10k. Per-epoch section losses are required and saved.
