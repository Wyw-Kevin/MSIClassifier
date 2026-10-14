# scRNA methods and evidence audit (2026-10-14)

Scope: read-only inspection and replay of saved scores/resampling indices. No classifier refitting, CNV rerun, model selection, or manuscript edits were performed. All new explicitly configured RNG seeds are 2026. Historical artifacts retain their target/model identity.

## 1. AUROC, cross-validation and bootstrap units

The main GSE178341 AUROC is **sample-level**, computed directly from **64 specimen OOF scores** (35 positive, 29 negative): **0.9783251231527094**. It is not calculated after aggregating scores to 62 patients. As an audit-only diagnostic, averaging each patient's scores first gives **0.9768907563025211**, based on 62 patients (34 positive, 28 negative); this is a different estimand and is not the manuscript's reported estimate.

Evidence:
- `最终模型/model_code/evaluate_inverse_weighted_mlh1.py:107–118`: reuse saved patient folds, assert patient separation, fit only training rows and write one score per specimen. Lines133–137 compute `roc_auc_score(table.y, table.score)` directly on each saved specimen table.
- `最终模型/model_code/prepare_weighted_scrna_article_results.py:51–68`: pivot by `sampleID`; observed AUROC uses `target` and specimen scores directly (line68). Lines54–63 draw patients within outcome strata and concatenate **all specimen indices** of each selected patient before calculating the bootstrap specimen AUROC. The point estimator remains specimen-weighted; the resampling unit is the patient.
- Fold origin: `最终模型/model_code/recompute_scrna_zscore_audits.py:19–31` constructs five folds using `make_patient_grouped_folds(..., random_state=2026)`; `最终模型/model_code/scrna_metadata_adjustment.py:393–419` uses `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026)` with patient IDs as groups. `最终模型/model_code/benchmark_scrna_shared18.py:30–37` writes the fold ledger reused by the weighted branch.
- Saved ledger: `最终模型/data/scRNA/result/diagnostics/shared18_same_sample_benchmark_seed2026/sample_ledger.csv`. C130TA/C130TB both belong to patient C130 and fold3; C171TA/C171TB both belong to C171 and fold2. Fold sample/patient counts are 12/12, 13/12, 14/13, 13/13, 12/12. Patient IDs never cross folds.

The newly added paragraph59 p-values have real saved evidence:
- Source `最终模型/report/article/第三版_中文版意见修订/prepare_evidence.py:33–62,64–82`: a single `np.random.default_rng(2026)` is initialized at line33; patients are sorted/encoded at43–46; each replicate draws positive and negative patient IDs with replacement at48–53; patient multiplicities are mapped to all specimen rows, then applied to the paired score-comparison kernel. Line61 calculates the two-sided, null-centered p-value `(1 + count(abs(delta_b - observed_delta) >= abs(observed_delta)-1e-12))/(B+1)`.
- Current workbook `最终模型/report/article/bulk12_loco_manuscript_seed2026_20261013/supplement/SupplementaryTablesS1-S14_English.xlsx`, sheet **S6_bootstrap**, matches `S6_Bootstrap_p.csv`.
- GSE178341: delta **0.0236453201970443**, p **0.1639180409795102**, 64 specimens/62 patients, 2000 replicates.
- GSE132465: delta **0.0789473684210526**, p **0.2173913043478260**, 23 specimens/23 patients, 2000 replicates.
- GSE200997: delta0, **no p-value and zero p-test replicates**; both joint models perfectly separate the two-positive cohort, so the paired delta is degenerate. Do not imply a 2000-replicate significance test was run for this cohort.
- Current `supplement/paired_patient_bootstrap_indices.npz` contains GSE178341 `(2000,62)` and GSE132465 `(2000,23)` patient-index arrays, corresponding patient dictionaries and paired deltas. Replaying the RNG across both cohorts reproduced **every saved draw exactly**, and recomputing all saved deltas yielded maximum difference0. GSE178341 resampled specimen counts vary from62 to70 because duplicate specimens travel together; each replicate always draws62 patient clusters.

Main AUROC confidence intervals and S7 intervals are separate archived bootstrap realizations. For the weighted GSE178341 joint model the main table's lower limit is **0.9390749299719888** (source `weighted_scRNA_update/AUROC_with_patient_bootstrap_CI.csv`), whereas S7's lower limit is **0.9380135360295097**. These correspond to different bootstrap loops/order, with identical point estimates; do not silently overwrite one with the other. The original main-CI/S7 scripts do not themselves save every draw, whereas the new p-test script does save the complete patient indices. Its “saved all indices” statement is supported specifically for the post hoc paired p-test.

Proposed methods clarification:
> GSE178341的AUROC直接基于64例肿瘤样本的折外评分计算，未先将同一患者的重复样本汇总。五折交叉验证以患者为分组单位，同一患者的样本固定进入同一折。置信区间及配对差值以患者为聚类重抽样单位，按结局分层抽取患者，并保留所抽患者的全部样本。事后显著性检验基于2,000次中心化配对差值分布，随机流以2026初始化一次并逐次推进；GSE200997因配对差值退化未计算p值。该推断以已保存评分为条件，未纳入上游特征发现、队列级CNV及模型重新拟合的不确定性。

## 2. CNV 100,000-cell sampling population

It is **all retained cells from tumor-tissue specimens**, across lineages; not malignant-only cells. Malignancy is classified later from CNV. Exact selection source: `最终模型/model_code/run_infercnvpy_malignant_reference.py:49–52,89–90,103–124`.

```python
rng = np.random.default_rng(SEED)                    # line90; SEED=2026 at28
keep, counts = sample_filter(obs)                   # line111; sample total>=200
a = a[keep].copy()                                  #112
tumor = a.obs['Class'].astype(str).eq('Tumor').to_numpy()  #115
ref_mask = a.obs['cnv_reference'].ne('').to_numpy()        #116
tumor_ids = np.flatnonzero(tumor)                    #117
ref_ids = np.flatnonzero(ref_mask)                   #118
if len(tumor_ids) > MAX_TUMOR:                       #119; MAX_TUMOR=100000
    tumor_ids = rng.choice(tumor_ids, MAX_TUMOR, replace=False) #120
if len(ref_ids) > MAX_REFERENCE:                    #121; MAX_REFERENCE=20000
    ref_ids = rng.choice(ref_ids, MAX_REFERENCE, replace=False) #122
selected = np.union1d(tumor_ids, ref_ids)             #123
a = a[selected].copy()                              #124
```

Metadata-only replay reproduced **the identical saved cell-ID set** of the current unique-cell stromal-only CNV result. After sampleQC, GSE178341 has257,251 tumor-specimen cells; 100,000 are drawn without replacement. The stromal reference contains15,436 cells and does not reach the20,000 cap. Union/deduplication yields **111,727** unique CNV cells, including **105,898** tumor-specimen cells and5,829 normal-specimen Strom cells. The tumor-specimen set comprises Epi44,085, B5,396, Mast1,014, Myeloid15,175, Plasma7,022, Strom9,607 andTNKILC23,599. The reference union therefore enriches Strom in the retained tumor subset; it is not a uniformly sampled final100,000-cell tumor set.

Saved manifest: `最终模型/data/scRNA/result/diagnostics/infercnvpy_stromal_only_unique_seed2026/GSE178341/manifest.json` records111,727 total analyzed cells,105,898 tumor cells,15,436 stromal reference cells, and seed2026. Its historical q99 output column does not define current weighted MLH1: the downstream feature branch explicitly recomputes q95 (see `mmr4_loss_vs_mlh1_001_unique2026/README.md:23–29`). Current candidate malignancy counts are unavailable for unsampled original cells; S1 correctly labels these as “not identified on unsampled cells.”

Proposed replacement for paragraph38:
> GSE178341在样本总细胞数≥200的QC后，从肿瘤组织样本来源的全部细胞中不放回抽取100,000个细胞，抽样对象包括上皮、免疫及基质细胞。基质参考细胞超过20,000时另行抽样。随机数生成器以2026初始化；肿瘤样本细胞索引与参考索引合并去重后进入CNV分析。最终保留111,727个细胞，其中105,898个来源于肿瘤组织样本；候选恶性上皮细胞在该保留集合中根据上皮注释及CNV规则后续确定。

## 3. Current S7 controls

`mu` is mean observed MLH1 in q95 candidate malignant epithelial cells; `p=N_candidate/N_all_epithelial` within the **same retained CNV cell set**; `W=mu*p`. “18+mu+p” is20 inputs, “18+p” and“18+W” are19 inputs; p-only is1 input. All controls use the fixed18 panel (when applicable), the same patient folds and training-fitted StandardScaler plus lsqr/auto-shrinkageLDA. The S7 controls were separately refitted as diagnostics; they are not alternative identities for the approved frozen artifact.

Exact control computation: `最终模型/report/article/revision_review_20260928/audit_evidence.py:71–105`, specifically configurations at76, patient separation78, training-fold fitting79, full-development external scoring81–85, and fixed-score sample-level AUROC100–102. Additional saved score source: `revision_review_20260928/additional_control_predictions.csv`; main four models: `weighted_scRNA_update/all_model_predictions.csv`. Current S7 CSV and workbook sheetS7 agree; replaying these saved scores reproduces the point estimates.

| Configuration | GSE178341 grouped OOF (64;35/29) | GSE132465 retrospective (23;4/19) | GSE200997 retrospective (15;2/13) |
|---|---:|---:|---:|
| p only | 0.8610837438 | 0.9671052632 | 0.5384615385 |
| 18 subtypes+p | 0.9674876847 | 0.9736842105 | 1.0000000000 |
| 18 subtypes+mu+p | 0.9566502463 | 0.9736842105 | 1.0000000000 |
| 18 subtypes+W (approved weighted joint) | 0.9783251232 | 0.9605263158 | 1.0000000000 |
| 18 subtypes+mu (raw malignant MLH1 joint) | 0.9546798030 | 0.8815789474 | 1.0000000000 |
| W only | 0.8926108374 | 0.9736842105 | 0.8846153846 |
| 18 subtypes only | 0.9615763547 | 0.8421052632 | 1.0000000000 |

Source `supplement/S7.csv` lines2–22 in the current bulk12 package; exact cell addresses are provided in `verification.json`. Active approved artifact: `最终模型/model/reference/shared18_weighted_mlh1_2026/model.joblib`; its manifest defines `mean_MLH1 * N_malignant / N_epi`, retains19 features, and its actual pipeline is StandardScaler + `LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto')`.

Proposed results addition:
> 仅候选恶性上皮比例模型在GSE178341患者分组OOF、GSE132465和GSE200997中的AUROC分别为0.8611、0.9671和0.5385；18亚型联合该比例的AUROC为0.9675、0.9737和1.0000。将候选恶性上皮MLH1均值与比例分开输入时，AUROC分别为0.9567、0.9737和1.0000；采用乘积加权MLH1时分别为0.9783、0.9605和1.0000。乘积形式在开发集的点估计较高，但外部队列并未显示一致增益，现有对照不能证明该形式具有稳定优势，亦不能仅据相关性确定MLH1的独立预测贡献。

## 4. MLH1 versus adding MSH2/MSH6/PMS2: evidence gap

**No preserved direct incremental MSI/dMMR comparison of the current18-subtype+weighted malignantMLH1 classifier against a corresponding model adding MSH2/MSH6/PMS2 was located.** The actual current classifier's19-feature list contains only the18 subtypes and its weightedMLH1 slot. Figure2E–H expression distributions are descriptive evidence and cannot establish a four-gene model's lack of incremental value.

A real related comparison exists, but has a different target:
- `最终模型/model_code/compare_mmr4_loss_mlh1_cutoff.py:25–46` reads four-gene means from the same unique stromal-onlyq95 candidate malignant cells.
- Lines47–53 define **local storedMLH1 loss**, with MSI+storedloss positive and MSS assignedexpress. This gives22 loss and42 express observations, not the35/29 MSI/dMMR outcome.
- Lines58–59 use StandardScaler+L2LogisticRegression(C=1,liblinear,unweighted,seed2026); lines69–87 fit patient-grouped folds and a separate final **MMR4_loss_model.joblib**. This model has **four unweighted gene means**, no18subtypes and no product-weighted inputs.
- `最终模型/data/scRNA/result/diagnostics/mmr4_loss_vs_mlh1_001_unique2026/training_metrics.csv:2–5`: over64 samples, negativeMLH1 AUROC **0.8712121212** versus four-geneOOF **0.8484848485**; within35MSI samples, **0.7360139860** versus **0.7237762238**. This is loss/express discrimination, not incremental MSI prediction. Older nonunique CNV results0.967532/0.962121 are historical and superseded for this corrected loss analysis; do not mix their identities.
- External assessment of that loss model is only MSS-screening flags:19 and13MSS samples, with1 and2 flagged by both methods; no externally confirmed MLH1-loss target exists.

Another preserved historical model is `benchmark_scrna_shared18.py:134–145`, “MMR4 expression(Logistic)”, using **whole-tumor pseudobulk log2CPM** rather than candidate malignant expression (pseudobulk construction89–96). It is a stand-alone four-gene expression comparator, not an18-subtype+MLH1 incremental model. Historical all-epithelialMLH1 panel/classifier comparisons also have a different feature definition and do not supply a current weighted four-gene increment test. No quantitative current weighted four-gene increment should be invented from these records.

Recommended paragraph35 amendment (keep the unverified present increment claim out):
> 本研究的既有单细胞实现保留MLH1作为MMR表达组分，并在当前模型中按上皮内CNV候选恶性细胞比例加权。MLH1、MSH2、MSH6和PMS2的样本级表达分布用于描述性比较。现有记录尚未提供在相同18亚型、患者折分及加权特征定义下，进一步加入其余三个基因的直接增量比较，因此不据此断言多基因方案不能改善当前MSI/dMMR模型。

Shorter manuscript-compatible version:
> 本研究沿用既有探索性实现中的MLH1表达组分，并按上皮内CNV候选恶性细胞比例加权；四个MMR基因的表达分布用于描述性比较。当前加权模型中加入MSH2、MSH6和PMS2的增量价值尚未进行同条件比较。

Paragraph87 should likewise omit “根据开发阶段的增量比较”, or explicitly scope it to a documented historical target/configuration. If citing the stored-loss analysis, state its different22/42 target, four-geneLogistic versus negativeMLH1 score and unweighted candidate-cell means; do not present it as19-versus22-feature MSI/dMMR model evidence.

## 5. Correlation/VIF evidence

The current official workbook already includes the new values in sheet **S7_correlation**, and they match `supplement/S7_Correlation_VIF.csv` and `第三版_中文版意见修订/weighted_feature_dependence.csv`. Source computation is `第三版_中文版意见修订/prepare_evidence.py:84–97` (correlation88–89; intercept-including least-squares VIF90–96). Recalculation from the saved S1 counts reproduced all18 correlation/VIF rows. No outcome labels enter these calculations.

GSE178341: Spearman W–mu=0.90151445193462, W–p=0.7711686428379663; VIF for W,mu,p=5.512565860104358,2.72896326350632,2.8807158881733934. Exact values are recorded in `verification.json` and delivered `current_S7_Correlation_VIF.csv`; manuscript-rounded values0.902/0.771 and5.513/2.729/2.881 are supported.

Delivered small supporting CSVs retain exact copied bytes and source hashes (`supporting_csv_manifest.json`). Full raw h5ad matrices and model artifacts were neither duplicated nor modified.
