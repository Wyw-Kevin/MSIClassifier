# Inputs for raw-data reconstruction

The package carries compact archived measurements, trained sample classifiers, reference summaries, posterior parameter means and run records. Full raw measurements and large learned annotation/reconstruction weights are listed here rather than copied. Exact original paths, sizes and available hashes are recorded in each modality provenance JSON; paths are historical identifiers, not mandatory installation locations.

| Branch | Additional material | Source of detailed inventory |
| --- | --- | --- |
| scRNA reference/migration | GSE178341, GSE132465 and GSE200997 input h5ad files with their original layers, clinical/sample metadata, and the correct frozen TOSICA weights | `code/scRNA/provenance_scRNA.json`, `reference_coverage.json`, and `README_scRNA.md` |
| scRNA CNV | Per-cohort raw/normalized cell matrices, gene coordinate mapping and retained stromal reference labels | scRNA manifest and original CNV scripts |
| BayesPrism | GSE178341.h5ad `layers['counts']` to regenerate the reference, plus each target's `bulk_data/<cohort>/expression.csv` | `code/bayesprism/provenance_bayesprism.json` and `README_bayesprism.md` |
| MMR4 log-input reconstruction | Stored log2 expression matrices for the current eight bulk cohorts; not BayesPrism's inverse-transformed integer matrices | `provenance/method_audits/audit_mmr4/`, supplemented by S12_expression_scale |
| Spatial reconstruction | Reference h5ad,14 original10x matrices and slice cell2location checkpoints for fresh posterior export/training | `code/spatial/provenance_spatial.json` and `README_spatial.md` |

Public study IDs are recorded in the manuscript and manifests. The exact preprocessed inputs used in the computation, including gene aggregation/filtering, must be respected; downloading an arbitrary current representation of a GEO series does not automatically reproduce the archived matrix.

Reference_seed2026 supports13/18 fixed-panel types. The five unsupported types are cTNI14,cTNI18,cTNI16,cTNI08,cTNI19. Earlier frozen ecological annotation weights support the full18 panel; retained dictionaries/masks are included, while the large weights remain external resources. These two reference families are not interchangeable. Missing measurements stop scoring rather than being silently zero-filled.

For current raw runs every explicitly configured seed/random_state is2026, negative deconvolution expression stops execution, classifier standardization is training-only, and no external outcome selects preprocessing, migration, features or fusion weights. See the archived user-approved protocols and modality README files for configuration details.
