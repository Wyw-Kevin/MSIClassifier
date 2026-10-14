# Bulk analysis identity

Current primary classifier: `classifier_snapshot/models/panel12_fold0.joblib` for full-development retrospective scoring; folds1–5 provide development OOF. All artifacts contain their actual features, training sample keys, parameters and seed2026. The source archive supplies17 candidates, development directional screening and the prior measured fractions.

MMR4 uses stored four-gene log2 inputs, cohort-specific ddof1 standardization and the exact eight held-out-cohort threshold locks. Its optimizer is in `mmr4/loco.py`. `mmr4/run.py` is the byte-preserved historical LOCO orchestration that also reports the older10-subtype classifier; it is not the current12-subtype default. Current primary scores are produced by `reproduce_frozen_bulk.py` using the current12 checkpoints and the same LOCO locks. Fusion is always0.5/0.5.

`paired_increment` preserves the exact paired analysis and its input copies; `compute_paired_increment.py --replay` recalculates intervals without fitting. `selection_bootstrap_source24` preserves the earlier24-candidate bootstrap records reused for the current17-candidate selection check; `compute_selection_stability.py` changes paths/output destination only, preserving RNG and computations. Its original is under `source_archive`.

Full-development feature selection and prior candidate direction definitions are fixed in classifier CV. BayesPrism measurements are cohort-adaptive. The 12-panel was adopted after comparative scores were examined. Results are conditional retrospective evaluations, not performance of a fully nested model-development process.
