# Sample-label and public-ID traceability

`data/S5_sample_label_sources.csv` covers1328 analyzed samples/sections plus the QC-excluded T_cac5. It preserves source-label fields/values and assay-confirmation gaps; spatial labels are public grouping inferences. S5 in the workbook remains the three discordant local-annotation cases.

`data/S5_renamed_GEO_ID_mapping.csv` maps81 GSE27544/GSE75316 analysis IDs to GSM and original matrix columns. Mapping is verified by unique maximum expression-value agreement across source probe-mean genes; `GEO_ID_expression_fingerprints.csv.gz` retains an exact-value witness. Original raw source file hashes and annotation hashes are recorded. Do not infer a missing public identifier from label agreement alone.

Ten GSE24550 original MSI-L records remain negative in the archived primary labels; `archived_MSI_L_negative_samples.csv` explicitly lists them. This revision does not relabel or recompute the primary models. Source assay details not present in stored metadata are marked unverified.

```bash
python workflows/traceability/verify_sources.py --output output/traceability.json
# Optional rebuild from the original raw project layout:
python workflows/traceability/rebuild_sources.py --project-root /path/to/raw_project --output-dir output/rebuilt_sources
```

The optional rebuild requires source clinical CSVs, GEO series matrices and GPL annotations under `bulk_data/`/`data/`, and the original GSE200997 clinical table under the archived final-model directory. Those large raw resources are not bundled. The saved-data verifier needs no raw project. In English source tables, `final_model/` is a display alias for the original project's final-model directory; this alias does not rename the original files.

The full-column mapping evidence requires >75% exact-value agreement across all comparable genes and more than tenfold separation from every alternate sample; the observed minimum agreement is96.77% for GSE27544 and95.68% for GSE75316. The128 retained witness genes per sample are a saved-value audit, not a new clinical feature panel.
