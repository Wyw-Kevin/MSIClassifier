# MSIClassifier

Local reproducibility package for the manuscript **A modality-specific transcriptomic framework for colorectal cancer MSI/dMMR prediction using shared cell subtypes and MMR expression**. This is an unpublished local snapshot, prepared on 14 October 2026; it is not a public release with a DOI. See [README_zh.md](README_zh.md) for Chinese instructions.

## Quick start: archived inputs and models

Use Python 3.12 in a separate environment, install `environment/requirements_core.txt`, and run from this folder:

```sh
python -m pip install -r environment/requirements_core.txt
python reproduce.py --branch all
```

The default runner performs CPU replays, bootstrap verification, a synthetic BayesPrism implementation check, and figure generation. It does **not** download raw data, fit classifiers, rerun CNV or train TOSICA/cell2location. Outputs go to `output/`; retained source artifacts are preserved. Individual choices are `bulk`, `scRNA`, `spatial`, `paired`, `stability`, `bayesprism`, and `figures`. Supply `--output-dir` to choose another location. A modern Linux installation with Cairo is needed for figure rendering; BayesPrism's synthetic check uses its recorded Linux implementation.

## Included material

| Path | Content |
| --- | --- |
| `supplement/MSIClassifier_补充方法与图片.docx` | Supplementary methods, figures S1–S3 and Chinese captions; no tables |
| `supplement/SupplementaryTablesS1-S14_English.xlsx` | English tables S1–S14 and expanded comparison, parameter and provenance sheets |
| `supplement/data_attachments/` | Machine-readable tables, specimen scores and saved resampling records |
| `code/scRNA/` | Actual preprocessing, annotation, CNV and model sources, historical reference dictionaries, archived features, fixed models, controls and replay |
| `code/bayesprism/` | Original and actual patched pybayesprism 0.1.0, complete helper modules, reference summaries, configs, final fractions and run records |
| `code/bulk/` | Current panel12/panel7 MLP artifacts, LOCO MMR4 sources/locks, paired increment and selection stability code, portable score replay |
| `code/spatial/` | Actual weighted spatial reconstruction and scoring sources, dependency closure, signatures, posterior means, abundance and spot records |
| `code/reporting/` | Actual plotting/build sources, editable publication SVGs and portable figure generation |
| `code/scRNA_sensitivity/` | Earlier capped sensitivity implementation, explicitly historical; current uncapped analysis is under `code/scRNA/` |
| `models/` | Preserved frozen scRNA classifier and manifest |
| `docs/` | Code availability, instructions, raw-data requirements and approved protocol copies |
| `environment/` | Tested replay dependencies and current runtime inventory |
| `provenance/` | Source hashes, document split audit, measurement audits and delivery verification |

## Model and measurement identities

- Primary scRNA: fixed18 subtype proportions plus `W = mean(candidate malignant epithelial MLH1) × N_malignant/N_all_epithelial`, training-only StandardScaler and shrinkage LDA. AUROC is specimen-level; CV and bootstrap group by patient. Saved current full model is replayed. Original primary CV fold objects were not saved; separately identified q95 sensitivity refits reproduce the archived OOF scores and retain their own identity.
- Primary bulk: 12 direction-concordant subtypes, MLP with one 4-unit hidden layer, training-only scaler; subtype five-fold OOF/full-development probabilities are fused with the corresponding eight-cohort-held-out MMR4 scores using fixed weights 0.5/0.5. **LOCO applies to MMR4 only.** The 12-panel was adopted after comparative results were viewed. Upstream reference directions and measurements remain fixed.
- Spatial: newly reconstructed proxy features enter the frozen scRNA classifier. This is technical transfer, not evidence that reconstructed and measured features are equivalent.
- Current archived ecology inputs were measured with an **older frozen TOSICA reference family** supporting the18 panel. The newer `reference_seed2026` family used for transferred Strom labels supports13/18 subtypes. Fresh annotation with that family cannot regenerate the full fixed18 panel. Missing types must not be replaced with zero.
- All explicitly configured seeds for current analyses are2026, with advancing resampling streams and no fold offsets. Historical model/source metadata retain their actual seeds; changing a file does not retrain a checkpoint.
- The CPU bundle verifies stored measurements, predictions and conditional analyses. Raw-data reconstruction needs the original expression matrices and large checkpoints listed in `docs/DATA_REQUIREMENTS.md`. It was not rerun for this packaging task.

## Further execution and provenance

See [docs/REPRODUCTION_GUIDE.md](docs/REPRODUCTION_GUIDE.md) before refitting or running raw-data branches. The complete actual BayesPrism implementation and its license are under `code/bayesprism/vendor/`; historical install commit and exact historical dependency snapshots were not retained. Current dependency versions are not substituted for those gaps. Spatial reference epoch-level training history was not recovered, so convergence is not asserted.

Third-party licenses remain with the corresponding code. This package does not assign a new license to the original project code. Before any public deposit, set its repository/archive version, access conditions and project license; no public publication was performed by this task.
