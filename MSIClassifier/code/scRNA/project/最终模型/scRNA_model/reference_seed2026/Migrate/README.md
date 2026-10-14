# Hierarchical TOSICA Migration Models

Reference: GSE178341 full single-cell h5ad, restricted to the common 16,496 protein-coding genes used by the existing reference.

Workflow:
1. Train one major model on `clTopLevel`.
2. Train subtype models for TNKILC, Myeloid, Strom, and Plasma.
3. Subtype models keep the subtype features used by the final scRNA model and merge all other cells in the same major class as `Other`.
4. Training uses balanced sampling to reduce majority-class domination.
5. A model is marked usable only if internal validation accuracy is greater than 0.95.

Usable models: 2/5

Key files:
- `tables/model_use_status.csv`
- `tables/eligible_subtypes_by_major.csv`
- `models/<model_name>/training_metrics.csv`
- `models/<model_name>/validation_predictions.csv`
- `models/<model_name>/validation_confusion_matrix.csv`
- `models/<model_name>/model_best.pth`

Model status:
            model_name  n_classes  best_val_acc  best_val_macro_recall  best_val_min_recall  best_val_auc_macro_ovr  usable
          tosica_major          7      0.995419               0.995354             0.989000                0.999492    True
 tosica_subtype_TNKILC          9      0.839018               0.799370             0.665511                0.974291   False
tosica_subtype_Myeloid          6      0.938264               0.832823             0.615741                0.981291   False
  tosica_subtype_Strom          5      0.978627               0.894363             0.740260                0.997714    True
 tosica_subtype_Plasma          2      0.940756               0.940883             0.939178                0.987548   False

Target subtype counts:
major
Myeloid    5
Plasma     1
Strom      4
TNKILC     8
