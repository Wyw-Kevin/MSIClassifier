# Spatial transfer replay

```sh
python workflows/spatial/reproduce_frozen_spatial.py --output-dir output/spatial --figures
```

This command reconstructs weighted MLH1 from saved candidate allocated-count/library totals, checks abundance against saved posterior means, reconstructs the18 parent-normalized proportions and verifies the14 section/7 patient frozen scores. It needs no GPU or new fitting. `--figures` calls the retained scientific `spatial()` function; Linux fontconfig supplies a font. Outputs are separate from resources.

Kept resources are only those actually read by this command: posterior means, abundance, spot totals, histories,73-type mapping, frozen feature/score tables and the ecology comparator. The main scRNA classifier is shared with `workflows/scRNA/`. The replay resources omit files unused by this command. For fresh mapping from raw10x/h5ad spot counts, run `python predict.py spatial` from the repository root after installing reference part2. The [expression workflow](../../docs/EXPRESSION_WORKFLOW.md) describes this route; it uses the separately supplied current73-type signature and saves new section posteriors and training history.

This preserves proxy-feature technical transfer, not measurement equivalence. The reference loss trajectory was not recovered and convergence is not asserted. The stored posterior summary is the parameter mean, and the candidate CP10K denominator spans the12079 filtered model genes. Spatial patient scores average two section probabilities.
