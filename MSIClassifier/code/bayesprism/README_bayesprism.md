# Current-paper BayesPrism code bank

This folder packages the actual `pybayesprism 0.1.0` local port used to produce the current paper's bulk subtype measurements. It copies code and archived results; it does not retrain weights or produce new cohort predictions. Current eight-cohort bulk12 input rows are a subset of the completed historical nine-cohort/1,270-sample branch. GSE92921 remains explicitly historical and is absent from the current eight-cohort/1,212-sample paper analysis.

## Contents and identities

- `vendor/`: the actual tested local modifications, original MIT license, package METADATA/RECORD and required gene-category files. Package author Zilu Wang, repository https://github.com/ziluwang829/pyBayesPrism; package version0.1.0. No historical Git commit was retained.
- `vendor_original/`: unmodified upstream files. Its default seed123 is retained as historical source provenance. Do not use it as the current paper implementation or relabel it seed2026.
- `historical_scripts/`: all original branch scripts copied byte for byte, including `patch_port.py` and `vendor_source_hashes.json`. The patch generator is one-time provenance, not an entry command to run against the already-patched vendor.
- Top-level Python files: portable copies of the actual scripts. Only root/output/interpreter paths, current cohort scope, output-report expected counts and export wording are adapted. Scientific code, reference rules, estimator settings and random-stream allocation are unchanged. `runtime.py` itself is byte-identical.
- `resources/reference/`: the full fixed reference-cell manifest (lossless gzip, around2.6MB), frozen72×17,039 filtered raw-count profiles (around10MB), type/parent mapping, gene audit and historical sample identities. Preparation uses the recorded `decode72_epi_seed2026` cell membership; it does not run DECODE. No DECODE helper module is required. The sample ledger originates from `decode70_bulk_audited2026`, but only dataset/sample IDs are used for BayesPrism preparation; outcomes do not choose reference/parameters.
- `resources/historical/`: original runtime metadata plus actual configs, completion metadata, saved phase-specific sample/gene identities and PCG64 states, final72-class posterior means and CV exports for all nine historical cohorts. The first eight listed below are current-paper cohorts; the ninth is preserved for historical closure.
- `resources/historical_exports/`: actual combined whole-reference/major/subtype-over-parent exports. These remain archived results, not results of executing portable scripts.
- `resources/optional_audit/`: small historical label/direction/model dependencies of the separate shared18 descriptive audit. That audit does not define the current bulk12 panel or choose weights from external AUROC.
- `provenance_bayesprism.json`: original absolute sources, byte hashes, decompressed cell-manifest hash, all portable adaptations, omitted large-data paths and package identity limits.

## Dependencies and verification

Use Python3.9 or newer for the portable validation helper. Execution is verified on Linux/POSIX; original fcntl locks and multiprocessing fork conventions are retained. Windows or spawn-only execution requires a separately reviewed platform adapter. The historical Python/dependency snapshot is unavailable; do not infer it from the current environment. `requirements.txt` lists direct third-party dependencies. `requirements_verified_environment.txt` records exact installed versions used for packaging verification, separately from historical runtime provenance. BayesPrism itself is imported from the included vendor; do not replace it with another package named `bayesprism` or a current repository checkout.

From the project directory:

```bash
python MSIClassifier/code/bayesprism/validate_portable.py
```

This verifies copied-resource hashes and imports, analytical conservation/gene filters, the archived abundance/parent normalization and a small two-type synthetic mixture. It does not compute any real cohort. The original `test_port.py` prints failures without raising; the validation wrapper invokes its tests directly so failures stop validation.

## Required external inputs

Large files are intentionally not duplicated:

1. `<MSI_PROJECT_ROOT>/GSE178341.h5ad` (public accession GSE178341), original `layers['counts']` and matching cell/gene identities. Source file was around18.4GB; no contemporaneous h5ad hash was retained. Its saved fixed cell manifest and prepared profiles are included. No per-cell total10,000 normalization or log transform is used in this BayesPrism reference.
2. `<MSI_PROJECT_ROOT>/bulk_data/<cohort>/expression.csv` for TCGA-COADREAD and GEO accessions GSE39582, GSE24550, GSE25071, GSE27544, GSE35896, GSE39084, GSE75316; optionally historical GSE92921. Exact saved source hashes and sizes are in provenance. These are historical gene-level platform matrices with the original gene mapping/duplicate aggregation already applied, not public raw probe files. GSE39582 preserves original-author ComBat. Downloading raw accession data alone does not reconstruct these processed matrix bytes.

TCGA's recorded log2(RSEM+1) is inverted with `2**x-1`; GEO processed log2 intensity is inverted with `2**x`; MAS5 GSE92921 remains linear. Nonfinite gene columns are excluded and recorded; negative selected input stops. Continuous linear input is saved as `*_linear.pkl`, but the **whole expression matrix is transformed with `np.rint(...).astype(np.int64)` before Prism.new**; the `*_sampling.pkl` integer matrix drives outlier filtering, initial inference, reference update and final inference. Half-integer ties go to the nearest even integer. Microarray intensity proxies are not RNA-seq read counts.

`prepare.py` reconstructs the reference and bulk inputs from the original h5ad and expression matrices. The complete frozen prepared reference is also bundled; `run_cohort.py` can read it directly once matching `*_linear.pkl` and `*_sampling.pkl` inputs exist in the new work directory. No separate bulk-only preparation entrypoint is introduced. Use a fresh work directory if inputs/configuration change: historical resume behavior does not independently revalidate existing completion markers against changed inputs.

## Entry commands and output paths

The following commands are examples for an **explicitly requested future rerun**. Packaging has not executed them on real data. Set paths before running; keep the work directory separate from historical results:

```bash
export MSI_PROJECT_ROOT=/path/to/MSI
export MSI_BAYESPRISM_WORKDIR=/path/to/new/bayesprism_rebuild
export MSI_BAYESPRISM_COHORTS=current8
python MSIClassifier/code/bayesprism/prepare.py
python MSIClassifier/code/bayesprism/launch.py
python MSIClassifier/code/bayesprism/postprocess.py
```

`launch.py` uses the currently selected Python interpreter and preserves the historical maximum3 concurrent cohorts /24 processes per cohort. For a serial explicit cohort run, use `python MSIClassifier/code/bayesprism/run_cohort.py GSE39582 24` after preparation. Set `MSI_BAYESPRISM_COHORTS=historical9` to recreate the original full nine-cohort scope, or give comma-separated unique supported IDs. GSE18088 is excluded because its VSN scale was not invertible as ordinary log2. `watch.py` is optional monitoring. `MSI_BAYESPRISM_EXPORT_DIR` can redirect exports; otherwise they go under the new work directory's `exports/`. The default work directory is this package's `runs/rebuild/`; archived resource directories are never output destinations.

## Scientific implementation and limitations

The reference is71 fixed non-Epi subtypes plus Epi. Profiles are raw-count sums over all retained fixed cells; no reference subsampling. After gene exclusions/protein-coding filtering and cohort-specific gene intersection, Prism normalizes each type's gene profile to one with pseudo_min1e-8. Bulk outlier cut/fraction is0.01/0.1. `key=None` treats all types equally; Epi is not candidate malignant epithelium. `update_gibbs=True` pools reference-update evidence within each cohort, then produces `posterior_theta_f.theta`. Each cohort begins from the same fixed reference. This is cohort-adaptive historical measurement, not frozen single-sample inference.

Actual configs: seed2026, alpha1, MAP sigma2, chain-length parameter1,000, burn500, thin2. Retained indices500,502,…,998 give250 posterior draws; the actual implemented loops execute999 updates per phase. The runtime allocates PCG64 stream blocks of2**64 from a root2026 generator; it does not add sample/fold/model offsets or reset the generator at each iteration. Copied upstream seeds outside these supported entry flows retain historical identities.

Final72-class posterior means sum to one within reference coverage. Parent abundance is the sum of the same final predictions sharing `clTopLevel`. The current bulk feature is subtype final fraction divided by its own major-parent sum, exported as `subtype_over_major.csv`. The stored final outputs, version/seed metadata and tests establish implementation traceability; they do not establish biological cell-fraction accuracy or cross-platform measurement equivalence.

Method: Chu T, Wang Z, Pe’er D, Danko CG. Nat Cancer. 2022;3(4):505–517. doi:10.1038/s43018-022-00356-3. Software: Zilu Wang, pyBayesPrism0.1.0 (2024). Local patches and source hashes are documented separately here. Historical code and observations must not be called newly untouched independent validation.
