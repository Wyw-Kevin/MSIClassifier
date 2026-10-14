# 当前 bulk12 的 BayesPrism 来源核查（2026-10-14）

本审计追踪当前 12 亚型模型实际使用的矩阵；未重跑反卷积、未重新拟合模型、未编辑稿件或共享代码。结论依据历史运行分支的保存资料，不以较新的探索实现替代其身份。

## 核实结论

当前模型使用 **pybayesprism 0.1.0 的存档本地修复版**，来源仓库为 [ziluwang829/pyBayesPrism](https://github.com/ziluwang829/pyBayesPrism)。包的作者为 Zilu Wang；该仓库描述的是 BayesPrism v2.2 的 Python 移植，0.1.0 是 Python 包的发行版本。[PyPI 发行资料](https://pypi.org/project/pybayesprism/)确认 0.1.0 于 2024-08-01 发行。没有保存可确证的安装 Git commit，不应填入当前仓库 HEAD。

当前 bulk12 丰度来源链为：

1. [/home/wyw/rstudio/MSI/最终模型/report/article/bulk12_loco_manuscript_seed2026_20261013/source_manifest.json](/home/wyw/rstudio/MSI/最终模型/report/article/bulk12_loco_manuscript_seed2026_20261013/source_manifest.json:11) 指向实际 7/12 比较模型目录。
2. [模型 source_manifest.json](/home/wyw/rstudio/MSI/最终模型/report/article/bulk_7_vs12_mlp_mmr4_loco_seed2026_20261013/source_manifest.json:9) 指向 `/home/wyw/rstudio/MSI/最终模型/report/article/pybayesprism9_seed2026/subtype_over_major.csv`，SHA256 为 `d1e5244a68fab27f6901efa45ec447b845f5b1e8cc3d9d37fbb1d38fd242141b`；当前原文件与 `source_archive/fractions.csv` 字节完全一致。
3. [postprocess.py](/home/wyw/rstudio/MSI/最终模型/data/bulk/result/pybayesprism9_seed2026/code/postprocess.py:9) 逐队列读取 `cohorts/<dataset>/final_fractions.csv`；第 12–13 行先按 `mapping.csv` 中的 `clTopLevel` 父类求和，再计算亚型/父类；第 19–20 行导出 `subtype_over_major.csv`。
4. [run_cohort.py](/home/wyw/rstudio/MSI/最终模型/data/bulk/result/pybayesprism9_seed2026/code/run_cohort.py:18) 将 `r.posterior_theta_f.theta` 写为 `final_fractions.csv`。这不是 `initial_fractions.csv`，不是 DECODE 的丰度，也不是未经更新参考的冻结单样本推断。

对九个完成队列重算导出公式，最大差异均为 1.11e-16；72 类丰度行和与 1 的最大偏差为 3.89e-15。当前八队列 1,212 行实际模型输入中，12 亚型与来源导出的最大绝对差为 4.44e-16。第九队列 GSE92921 保留在原始九队列分支中，但未进入当前八队列 bulk12 评价。

## 版本与本地改动证据

- [METADATA](/home/wyw/rstudio/MSI/最终模型/data/bulk/result/pybayesprism9_seed2026/code/vendor/pybayesprism-0.1.0.dist-info/METADATA:2) 保存包名、0.1.0、Zilu Wang 和仓库网址。
- [run_cohort.py](/home/wyw/rstudio/MSI/最终模型/data/bulk/result/pybayesprism9_seed2026/code/run_cohort.py:4) 把本分支 `code/vendor` 放在导入路径首位；第 12–17 行设置 2026 并构建/运行 Prism；各队列 `config.json` 第 87 行记录 `pybayesprism0.1.0 with documented audited fixes`。
- [patch_port.py](/home/wyw/rstudio/MSI/最终模型/data/bulk/result/pybayesprism9_seed2026/code/patch_port.py:10) 记录采样循环端点、基因类别和未匹配基因过滤、最终输出列标签、优化目标汇总与控制字典副作用修复，以及 NumPy Generator/向量化多项分布采样。原始源码保持在 `vendor_original`，默认历史 seed123 仍是原包身份，未改写或重标。
- [runtime.py](/home/wyw/rstudio/MSI/最终模型/data/bulk/result/pybayesprism9_seed2026/code/runtime.py:23) 在运行时覆盖 `GibbsSampler.run_gibbs_refPhi`。以 2026 初始化 PCG64，持续分配 2**64 状态区段，保存样本随机状态；初始与最终阶段使用推进的状态分配器，不重置每次迭代。
- 本次核验：24 项 `vendor_source_hashes.json` 记录均匹配；9 项 `validation.json` 脚本哈希均匹配；12 个 `vendor_original` Python 文件均匹配安装 `RECORD`。
- [launch.py](/home/wyw/rstudio/MSI/最终模型/data/bulk/result/pybayesprism9_seed2026/code/launch.py:11) 记录 Python 可执行文件 `/home/wyw/miniconda3/envs/MSI/bin/python`。本分支未找到当时 Python/NumPy/SciPy/pandas 的精确版本快照，不可借用后来分类器的版本信息代填。

## 参考表达层

[prepare.py](/home/wyw/rstudio/MSI/最终模型/data/bulk/result/pybayesprism9_seed2026/code/prepare.py:8) 使用 `decode72_epi_seed2026/reference/cell_manifest.csv` 的固定细胞集合，第 10 行指向 `/home/wyw/rstudio/MSI/GSE178341.h5ad`，第 18 行实际读取 **`layers['counts']`**，第 21–27 行按类型汇总原始计数并过滤基因。

保存参考有 257,129 个细胞（Epi 113,216 个）、71 个非上皮亚型及 1 个 Epi；非 Epi 亚型最少 53 个细胞。`raw_count_profiles.pkl` 为 72×17,039，全部值有限、非负且为整数值。这里没有逐细胞 total10000 归一化或 log1p。重复基因符号先求和；保留至少 5 个参考细胞表达的基因，排除 Rb/Mrp/other_Rb/chrM/MALAT1/chrX/chrY，再保留 protein_coding。

[Prism.new](/home/wyw/rstudio/MSI/最终模型/data/bulk/result/pybayesprism9_seed2026/code/vendor/pybayesprism/prism.py:113) 先按 bulk 默认规则剔除 outlier 基因（0.01/0.1），再匹配参考/目标基因；第 134–135 行在队列交集上将每一类型的参考基因谱归一化至总和 1（pseudo_min=1e-8）。八个当前队列最终运行基因数依次为 GSE24550 13,616；GSE25071 9,505；GSE27544 14,348；GSE35896/GSE39084/GSE39582 均 15,082；GSE75316 15,319；TCGA-COADREAD 14,794。注意 TCGA 初始 overlap 为 14,795，outlier 过滤后实际为 14,794。

当前 h5ad counts 编码为 CSR、形状 370,115×43,113、存储 dtype=float64；抽查前 100,000 个存储值为非负有限整数值。源 h5ad 当时的文件哈希未记录，所以这可验证源路径/层和保存的参考，但不能证明当前 h5ad 与历史运行时逐字节一致。保存参考 SHA256：`341edbf7cf085be2c4312c8d560e9a29091a332ec1970ca8df16b35bbeb0685c`。

## 取整实际影响

**实际调用为 `np.rint`，不是 `np.round`；并非仅对抽样时的临时总量取整。**

[prepare.py](/home/wyw/rstudio/MSI/最终模型/data/bulk/result/pybayesprism9_seed2026/code/prepare.py:38) 先按队列作线性尺度恢复并保存 `*_linear.pkl`；第 44 行对整张线性矩阵逐项执行 `np.rint(mat).astype(np.int64)`，保存为 `*_sampling.pkl`。[run_cohort.py](/home/wyw/rstudio/MSI/最终模型/data/bulk/result/pybayesprism9_seed2026/code/run_cohort.py:13) 明确读取后者送入 `Prism.new`。因此取整影响 outlier 过滤、初始 Gibbs 推断、基于初始分配的 MAP 参考更新及最终 Gibbs 推断；连续版被存档但没有送入这次正式模型。参考原始 counts 没有在这里再取整。

[NumPy 官方 rint 文档](https://numpy.org/doc/stable/reference/generated/numpy.rint.html)说明，半整数取最近偶数；泛称“传统四舍五入”可能不精确。[gibbs.py](/home/wyw/rstudio/MSI/最终模型/data/bulk/result/pybayesprism9_seed2026/code/vendor/pybayesprism/gibbs.py:90) 与第 133 行只将已有整数输入再次转换为 int64 送入多项分布。实际检查 GSE39582 与 TCGA：sampling 完全等于 linear 的 `np.rint(...).astype(np.int64)`，aligned_mixture 完全等于 sampling 的运行基因子集。最大逐项误差分别为 0.4999999873 与 0.4999966507。芯片强度代理不能因此解释为 RNA-seq 原始读数。

## 参数和最终输出

`key=None`，`update_gibbs=True`，链长度参数 1,000、burn-in500、thin2、alpha1、MAP sigma2；各队列独立用同一初始参考，按本队列初始分配更新参考。[optim.py](/home/wyw/rstudio/MSI/最终模型/data/bulk/result/pybayesprism9_seed2026/code/vendor/pybayesprism/optim.py:185) 的 key=None 分支对本队列样本汇总并更新所有类型，无恶性专属参考。因此 Epi 不等同于候选恶性上皮。

[gibbs.py](/home/wyw/rstudio/MSI/最终模型/data/bulk/result/pybayesprism9_seed2026/code/vendor/pybayesprism/gibbs.py:47) 实际保留索引 500,502,…,998，共 250 个后验样本；循环第 83/130 行执行 `range(max_idx+1)`，即 0…998，共 999 次更新。正文可写“链长度参数设为1,000”；不宜写成已实际执行1,000次。这个精确性注释不要求重跑或改变结果。

最终矩阵为各队列 `posterior_theta_f.theta`/`final_fractions.csv`，全 72 类为参考覆盖范围内的相对丰度。父类为同一最终矩阵中共享 `clTopLevel` 的亚型之和，分类器使用 `subtype_fraction / parent_sum`，而不是除以全组织丰度总和，也不是中层类分母。

## 文献及稿件替换建议

当前 `source_inventory.json` 中 [32] 为 DECODE 文献，正文未见 `[32]` 引用。建议将第 32 条替换为以下方法论文，在 BayesPrism 方法段引用 [32]；追加软件来源为第 37 条并在实际版本说明中引用 [37]，保持 33–36 编号不变。

**32.** Chu T, Wang Z, Pe’er D, Danko CG. Cell type and gene expression deconvolution with BayesPrism enables Bayesian integrative analysis across bulk and single-cell RNA sequencing in oncology. Nat Cancer. 2022;3(4):505–517. doi:10.1038/s43018-022-00356-3. PMID:35469013. [论文](https://www.nature.com/articles/s43018-022-00356-3)；[PubMed](https://pubmed.ncbi.nlm.nih.gov/35469013/)。

**37.** Wang Z. pyBayesPrism: Python implementation of BayesPrism. Version 0.1.0 [Internet]. 2024 [cited 2026 Oct 14]. Available from: https://github.com/ziluwang829/pyBayesPrism. [实现仓库](https://github.com/ziluwang829/pyBayesPrism)；[包发行记录](https://pypi.org/project/pybayesprism/)。此 Wang 为软件作者 **Zilu Wang**；方法论文中的 Wang 为 **Zhong Wang**。这是软件引用，不是另行确证的同行评议 Python 移植论文，不能赋予虚构 DOI/commit。

可用于方法的三句：

两个开发队列与六个回顾性GEO队列分别采用BayesPrism[32]的Python移植pybayesprism 0.1.0[37]（含存档本地修复），以GSE178341.h5ad的原始counts层构建由71个非上皮亚型和Epi组成的固定参考。 连续线性表达予以保存，运行前逐项使用NumPy rint取最近整数（半整数取最近偶数）并转为int64，取整矩阵用于初始Gibbs推断、队列内参考更新及最终丰度估计。 提取更新参考后的后验平均相对丰度，再以各亚型丰度除以同一样本所属主要大类的丰度总和，得到bulk分类器输入。

原段其余基因覆盖和参数可保留，并将“运行1,000次迭代”调整为“链长度参数设为1,000”。连续表达/取整句应替换到输入段，不应与“连续输入直接用于模型”并存。
