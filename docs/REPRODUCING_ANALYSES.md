# Reproduce paper analyses

[User guide](../README.md) · [Advanced commands](TUTORIAL.md) · [Parameter reference](PARAMETERS.md)

This page covers the archived-input analysis commands. For prediction from a new expression matrix, follow [EXPRESSION_WORKFLOW.md](EXPRESSION_WORKFLOW.md).

## Run archived analyses

```bash
python reproduce.py --branch all
python reproduce.py --branch all --plots --output-dir output/paper_replay
```

The dispatcher uses bundled measurements and saved classifiers. It does not rerun annotation, CNV, raw spatial fitting or classifier training. The BayesPrism verification branch checks its reference and a synthetic mixture.

| Branch | Computation |
| --- | --- |
| `bulk` | Five-fold development and saved full-fit retrospective scores |
| `scRNA` | Saved classifier predictions, patient-grouped evaluation and composition controls |
| `sensitivity` | Stored subtype-selection and CNV-threshold model checks |
| `spatial` | Proxy features and section/patient scores from saved spatial outputs |
| `paired` | Paired fusion-minus-MMR4 AUROC differences and intervals |
| `stability` | Archived development resampling and candidate selection frequencies |
| `bayesprism` | Patched implementation, reference and synthetic-mixture checks |
| `plots` | Figures from the retained numerical results |

Each branch also runs individually, for example:

```bash
python reproduce.py --branch paired
python reproduce.py --branch stability
python reproduce.py --branch spatial --plots
```

Logs and a completion summary are written beneath the selected output directory. Original measurements and models remain unchanged.

## Refit or rebuild upstream measurements

Optional bulk/scRNA classifier refits and the historical raw-cohort BayesPrism preparation commands are listed in [TUTORIAL.md](TUTORIAL.md). The expression entry `predict.py` applies the current classifiers to newly generated measurements.

The archived development bulk scores use fold1–5 models; direct bulk prediction uses the complete-development fold0 model. Those two outputs serve different computational roles. The current17 scRNA primary stores all five training-fold objects and the full-development classifier under a distinct model identity; both reproduce the archived17 threshold results.

The eight MMR4 parameter files belong to cohort-specific evaluation rounds. Threshold fitting and fixed-score replay are separate commands. The paired comparison uses the same samples and resampling indices for both branches.

## Model and processing records

Exact feature orders, model hashes, folds, reference sources and RNG records are retained with the executable analyses and in [provenance](../provenance). Historical training seeds remain attached to their original weights. All new explicit seeds use2026.

Frozen ecological annotation uses the older Major/CRC family that supports17/17 required classifier types. The separate2026 Major supplies external Strom/Epi calls for CNV. Spatial replay and new spatial prediction both use reconstructed proxy features; the stored reference regression has no retained epoch history. Detailed computational definitions remain in the workflow and parameter documents.
