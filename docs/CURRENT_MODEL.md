# Current model version

scRNA: all17 subtypes with GSE178341 orientedAUROC>0.70, plus weighted MLH1, training-only Z-score and shrinkage LDA. Model and manifest: `workflows/scRNA/resources/model/reference/auc070_17_weighted_mlh1_2026/`.

| Source cutoff | Subtypes | Development patient-grouped OOF AUROC |
| --- | ---: | ---: |
| >0.60 |39 |0.9349753695 |
| >0.65 |24 |0.9546798030 |
| >0.70 |17 |0.9802955665 |
| >0.75 |9 |0.9625615764 |

The final cutoff was selected after this development comparison. Source screening is not repeated within folds and the same OOF result selects the cutoff; bootstrap intervals are conditional on fixed scores and exclude these selection steps. External-evaluation AUROCs are0.9473684211 (GSE132465,23 samples/4 positive) and1.0000 (GSE200997,15 samples/2 positive). The current scRNA subtype screening, cutoff selection, scaler fitting and classifier fitting use GSE178341 only. GSE132465 and GSE200997 are external evaluation cohorts: their samples and outcome labels are not used to construct the current model. Both use the frozen full-development scaler and classifier. Development OOF performance remains conditional on using the same OOF comparison to select the cutoff; it is not nested internal validation.

Bulk remains12 subtypes plus leave-one-cohort MMR4, fixed0.5:0.5. Its original17-candidate pool is identical to the new scRNA panel, so no bulk feature/model refit is needed. Spatial uses this frozen17 classifier with the same73-type reference and reconstructed proxy inputs. Existing reference resource parts1/2 are unchanged. Historical18 resources remain separately named for provenance.

Frozen ecology and CNV annotation models have distinct roles: see [model-reference map](MODEL_REFERENCES.md). CNV quantile ties can yield identical numeric thresholds. The [boundary replay](../workflows/scRNA/boundary/README.md) compares equality inclusion with strict >q95 using the frozen classifier; zero candidate cells remain missing. See [analysis protocol scopes](ANALYSIS_PROTOCOLS.md) and [sample traceability](../workflows/traceability/README.md).
