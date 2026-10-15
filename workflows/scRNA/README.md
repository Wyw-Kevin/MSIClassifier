# Single-cell scoring and reproduction

The current model uses **all17 source-oriented AUROC>0.70 subtypes plus weighted MLH1**. `predict_scrna.py` scores finite sample rows with the distinct `auc070_17_weighted_mlh1_2026/model.joblib` artifact. Start from expression using the root `predict.py scrna` command; see [input files](../../docs/INPUT_FILES.md) and [expression tutorial](../../docs/EXPRESSION_WORKFLOW.md). Existing reference resources1/2 remain compatible.

```bash
python workflows/scRNA/predict_scrna.py --feature-table examples/scRNA_features.csv --output output/scRNA_scores.csv
python reproduce.py --branch scRNA --output-dir output
python reproduce.py --branch sensitivity --output-dir output
python workflows/scRNA/primary17/code/run_analysis.py
python reproduce.py --branch plots --output-dir output
```

The first command is unlabeled inference. Replay verifies all stored fold/full predictions and patient-bootstrap intervals independently with sklearn ROC. `run_analysis.py` explicitly refits fold/full classifiers and controls from bundled measured features; it does not rerun annotation or CNV. Explicit seeds are2026 and resamples advance one RNG stream. Optional refitting writes under `primary17/models` and `primary17/tables`; copy the package first if preserving the supplied snapshot.

The source direction and single-feature AUROC use full development data. Four cutoffs0.60/0.65/0.70/0.75 select39/24/17/9 subtypes. Cutoff0.70 was adopted after observing the highest development groupedOOF AUROC0.9803. The same OOF comparison selects the cutoff; it is not nested evaluation and its conditional CI excludes selection uncertainty. The current model is constructed only in GSE178341. GSE132465 and GSE200997 provide external evaluation using the frozen full-development scaler/classifier; their samples and labels do not enter current feature or cutoff selection or classifier fitting. Historical18 resources retain their original identity and are not the default model. Current spatial scores use17 features with the unchanged73-type reconstruction.

To refit all four cutoff configurations from measured features: `python workflows/scRNA/primary17/code/run_cutoff_comparison.py`. To verify the supplied24 cutoff models and all three-cohort AUROC/BA/Brier results without refitting: `python reproduce.py --branch sensitivity --output-dir output`.
