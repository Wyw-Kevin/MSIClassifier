# scRNA

`predict_scrna.py` applies the frozen fixed18＋weighted-MLH1 model without labels. Supply `sampleID`, every subtype column named in `resources/model/reference/shared18_weighted_mlh1_2026/manifest.json`, and explicitly weighted MLH1. `--model` can select a matching artifact, and `--weighted-mlh1-column` can select a verified weighted column. Missing or non-finite values stop execution.

`replay_frozen_scRNA.py` verifies the64 apparent＋23＋15 frozen scores, archived patient-grouped OOF,21 S7 control AUROCs, saved patient bootstrap and rho/VIF. Original primary CV-fold objects were not saved; separately identified q95 sensitivity fold refits reproduce stored OOF scores. Those model identities remain separate.

Current sensitivity code is under `resources/article/scRNA_AUROC_threshold_all_candidates_seed2026_20261013`. Verification reads saved models/inputs and writes a separate result:

```sh
python reproduce.py --branch sensitivity
```

Explicit optional refit from the retained source feature measurements:

```sh
python workflows/scRNA/resources/article/scRNA_AUROC_threshold_all_candidates_seed2026_20261013/run_analysis.py --output-dir output/scRNA_sensitivity_refit
python workflows/scRNA/resources/article/scRNA_AUROC_threshold_all_candidates_seed2026_20261013/verify_analysis.py --analysis-dir output/scRNA_sensitivity_refit --output output/scRNA_sensitivity_refit_checks.json
```

The uncapped thresholds select all39/24/17/9 passing candidates. The fixed18 historical reference is separate; CNV q90/q95/q99 comparisons use fixed retained measurements and identical patient folds. These optional commands refit classifiers, not CNV/TOSICA. Runtime remains2026 where a seed is configured. The current scalar isW=μ×N_candidate/N_all_epithelial; zero candidates mean missing. No50-cell sample rule applies.

For a new h5ad/10x expression dataset, use `python predict.py scrna` from the repository root. The [expression workflow](../../docs/EXPRESSION_WORKFLOW.md) documents input metadata, scales and installation of both frozen-reference archives. This route performs annotation, Strom-only CNVq95, current weightedMLH1 construction and frozen scoring. The feature-table command above skips those upstream steps.
