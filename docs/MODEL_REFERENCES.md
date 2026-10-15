# Frozen annotation references and feature outputs

| Purpose | Frozen model family | Output |
| --- | --- | --- |
| Ecology parent labels | `resources/raw_reference/ecology/tosica_major` | Major labels used with the historical subtype models |
| Ecology subtype labels | `resources/raw_reference/ecology/CRC/{TNKILC,Myeloid,Strom,Plasma}` | All17 within-parent subtype features |
| External CNV reference labels | `resources/raw_reference/cnv_major` from `reference_seed2026` | Strom labels for CNV reference selection; not a replacement ecology subtype model |

GSE178341 uses its original source annotations. Exact checkpoint files, SHA256 values, label dictionaries, gene orders and masks are listed in [the model-feature map](../provenance/TOSICA_checkpoint_feature_map.csv). The newer full hierarchical reference family supports12/17 required ecology types, missing cTNI14/cTNI18/cTNI16/cTNI08/cTNI19; its subtype outputs are not used for the current ecology inputs. This coverage limit does not mean five current classifier inputs are missing. Preserve historical checkpoint seeds and provenance.

Reference resource archives1/2 are unchanged; no annotation or CNV rerun is required for this documentation update.
