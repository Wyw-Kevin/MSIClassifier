# Frozen CNV boundary-rule diagnostic

```bash
python workflows/scRNA/boundary/replay_boundary.py --output-dir output/boundary
```

Uses saved primary candidate-cell MLH1, exact retained CNV scores, source denominators and the current17 q95 fold/full models. Compare the primary `score>=q95-1e-12` with strict `score>q95`; all ecological inputs and epithelial denominators remain fixed. This is input perturbation, not classifier/CNV refitting or external-based threshold selection. Zero candidates means missing, never an imputed zero; AUROC/BA comparisons use common scoreable specimens.

GSE200997 loses1 scoreable specimen (T_cac13). On the common14, AUROC remains1.000 but BA changes0.9583 to0.8750; maximum score change is0.8092. Thus unchanged AUROC does not establish unchanged W, scores or fixed-cutoff classifications. The primary q95 definition is unchanged.
