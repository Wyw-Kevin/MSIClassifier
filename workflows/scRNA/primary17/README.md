# Current17 analysis snapshot

`inputs` preserves the exact development/external-evaluation features, source screening and archived17 checkpoints. `models` contains newly refitted fold/full primary, ablation and CNV-sensitivity classifiers. `tables` contains actual scores, metrics and patient draws. `spatial` contains current17 scores from the unchanged73-type reconstruction. The active full classifier is copied to the explicit17 deployment directory. Current17 and historical18 identities are distinct.

Use `code/verify_primary17.py` for frozen replay, `code/run_analysis.py` for explicit refitting from measured features, and the plotting scripts for numerical figures. No raw TOSICA/CNV/cell2location fitting is implied by this snapshot. The source SVGs are layout templates; their old labels are replaced by the plotting program using current numeric results.

To refit all four cutoff configurations from measured features: `python workflows/scRNA/primary17/code/run_cutoff_comparison.py`. To verify the supplied24 cutoff models and all three-cohort AUROC/BA/Brier results without refitting: `python reproduce.py --branch sensitivity --output-dir output`.
