# MSIClassifier 使用说明

MSIClassifier用于结直肠癌MSI/dMMR研究评分，将固定细胞亚型组成与MMR相关表达信息结合。仓库提供冻结模型评分和论文既有分析复现两类流程。直接评分不需要BAM或MSI标签，但输入必须是按对应定义构建的特征表；普通基因表达矩阵不能直接替代亚型比例和加权MLH1。

本仓库通过Python脚本运行，目前未提供PyPI命令或Docker镜像。正文、补充表图、大型原始数据和历史来源另行归档。[English README](README.md)。

## 一、提供哪些模型

- 单细胞：18个亚型比例＋加权MLH1，应用训练集拟合并冻结的StandardScaler和收缩LDA，输出样本级 `score`。
- bulk亚型分支：12个父类归一化亚型丰度，应用完整开发集StandardScaler和4单元MLP，输出 `Subtype`。
- 已有队列bulk融合：额外提供四基因原始对数表达及匹配MMR4参数锁，输出 `MMR4_LOCO` 和固定0.5∶0.5的 `Fusion05`。
- 空间：从已保存的spot分配计数、丰度及后验均值复现代理特征与冻结分类器评分，覆盖14张切片和7位患者。

加权MLH1定义为：

~~~text
weighted_MLH1 = 候选恶性上皮细胞的MLH1平均表达
                × 候选恶性上皮细胞数 / 全部上皮细胞数
~~~

两项计数来自同一CNV保留集合。细胞表达采用总量10,000归一化后的自然对数log1p尺度。无候选细胞时为缺失，不能填0；有候选细胞且实测MLH1为0时保留。该特征不能用全上皮MLH1均值替代。

八组MMR4锁分别对应八轮留一队列评价，不能任意选一组当作所有新队列的统一部署模型。未知新队列可以先使用bulk亚型分支；融合需要另行确定合适的阈值和标准化协议。

## 二、安装

已经核对的环境为Linux和Python3.12，依赖版本列于 `requirements.txt`。直接评分及已保存输入的复现不需要GPU。BayesPrism辅助程序使用Linux进程和文件锁；空间绘图还需要fontconfig。

解压后进入 `MSIClassifier`，以下命令都在该目录执行：

~~~bash
cd MSIClassifier
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
~~~

也可使用Conda：

~~~bash
conda create -n msiclassifier python=3.12 pip
conda activate msiclassifier
python -m pip install -r requirements.txt
~~~

Ubuntu/Debian系统如缺少 `fc-match`，仅在需要空间成对绘图时安装：

~~~bash
sudo apt-get install fontconfig
~~~

查看命令帮助：

~~~bash
python reproduce.py --help
python workflows/scRNA/predict_scrna.py --help
python workflows/bulk/predict_bulk.py --help
~~~

BayesPrism使用仓库内保存的实际修复版，不通过额外安装另一个Python移植包替换它。

## 三、先运行示例

### 单细胞示例

~~~bash
python workflows/scRNA/predict_scrna.py \
  --feature-table examples/scRNA_features.csv \
  --output output/scRNA_scores.csv
~~~

示例包含5个样本和完整19维输入。输出前两例应约为0.119320418和0.083181782。[完整预期结果](examples/scRNA_expected_scores.csv)。

### bulk亚型示例

~~~bash
python workflows/bulk/predict_bulk.py \
  --feature-table examples/bulk_features.csv \
  --output output/bulk_scores.csv
~~~

此处只运行12亚型分支，不需要结局标签或MMR4。[预期结果](examples/bulk_expected_scores.csv)。

### GSE24550固定融合示例

~~~bash
python workflows/bulk/predict_bulk.py \
  --feature-table examples/bulk_features.csv \
  --mmr4-lock workflows/bulk/mmr4/locks/heldout_GSE24550.json \
  --output output/bulk_fusion.csv
~~~

示例来自GSE24550，使用该队列已保存的均值、样本标准差和阈值，不对5例示例重新计算Z-score。[完整预期结果](examples/bulk_expected_fusion.csv)。

## 四、准备自己的输入文件

CSV必须包含表头，**每行一个样本，每列一个特征**。列名区分大小写，使用 `sampleID`。数值应为实测原尺度；模型内部已经应用训练标准化，不要提前再做Z-score。

单细胞需唯一的 `sampleID`、以下18个亚型和 `weighted_MLH1`：

~~~text
cTNI14, cTNI18, cS18, cTNI13, cTNI16, cTNI05, cTNI07, cS27,
cTNI04, cS17, cM01, cP1, cM05, cM06, cTNI08, cTNI19, cTNI10, cM04
~~~

bulk需样本标识和以下12个亚型：

~~~text
cS27, cTNI16, cM06, cTNI05, cTNI10, cP1,
cTNI13, cS18, cTNI04, cS17, cM05, cTNI08
~~~

亚型值是占所属大类的比例，范围0–1。cTNI的分母为TNKILC，cS为Strom，cM为Myeloid，cP1为Plasma；不能用占全部细胞的比例、0–100百分数或经过方向乘号变换的数值。[特征与父类清单](examples/required_features.csv)。

融合另需匹配的 `dataset`、JSON参数锁，以及 `MLH1_raw`、`MSH2_raw`、`MSH6_raw`、`PMS2_raw`。这些列虽名为raw，实际是标准化前的对数表达：TCGA为log2(RSEM+1)，GEO为处理后的log2微阵列表达，不是BayesPrism反变换或取整后的数据。

替换输入路径即可评分：

~~~bash
python workflows/scRNA/predict_scrna.py \
  --feature-table my_scRNA_features.csv \
  --output output/my_scRNA_scores.csv
~~~

如果存档列名为 `mean_MLH1_epithelial`，且确认其中存的是当前加权特征，可以显式加：

~~~text
--weighted-mlh1-column mean_MLH1_epithelial
~~~

这不会使旧版全上皮或未加权MLH1自动变成可用输入。不能用0填补未测量的亚型，也不能将新版参考缺少的5个亚型当作不存在。原始细胞注释/CNV和原始空间重建属于上游流程，紧凑仓库的直接评分程序不代替这些步骤。

## 五、命令参数

单细胞评分：

- `--feature-table`：必填，输入CSV路径。
- `--output`：必填，输出CSV路径；程序自动创建上级目录。
- `--model`：可选，默认使用随附冻结模型。
- `--weighted-mlh1-column`：可选，默认 `weighted_MLH1`。
- `--help`：显示帮助并退出。

bulk评分：

- `--feature-table`、`--output`：必填。
- `--model`：默认完整开发集的 `panel12_fold0.joblib`；不能用OOF折模型冒充此入口的完整模型。
- `--mmr4-lock`：默认不提供；提供匹配队列JSON时计算MMR4和固定融合。
- `--help`：显示帮助。

复现入口：

- `--branch`：默认 `all`；可选 `bulk`、`scRNA`、`sensitivity`、`spatial`、`paired`、`stability`、`bayesprism`、`plots`。
- `--output-dir`：默认仓库下的 `output/`。
- `--plots`：同时生成科研图，空间分支也生成成对点图。

融合权重、MMR4温度和已冻结特征面板不属于预测命令的可调参数。[完整参数、模型配置和环境变量说明](docs/PARAMETERS.md)。

## 六、复现已有分析与生成图

~~~bash
python reproduce.py --branch all
python reproduce.py --branch all --plots
python reproduce.py --branch paired
python reproduce.py --branch stability
~~~

默认流程使用归档测量和模型，不重新训练或运行原始TOSICA/CNV/cell2location。BayesPrism分支进行参考与合成样本检查，不重算真实队列。输出包括日志、CSV和验证JSON。

bulk直接评分采用完整开发集模型，因此对开发样本是表观评分；论文开发集OOF结果要用复现入口中的五折模型。单细胞原始主分析五折对象未保存，另存q95敏感性重拟合模型可数值复现OOF，但保持其独立拟合身份。

可选重拟合、真实队列BayesPrism运行、完整所需原始文件及worker参数在[逐步教程](docs/TUTORIAL.md)中单独列出，不由 `--branch all` 自动执行。

## 七、输出和结果含义

单细胞输出样本标识与 `score`；bulk输出 `Subtype`，使用参数锁时另含 `MMR4_LOCO` 和 `Fusion05`。数值越高越倾向研究中的阳性组，但它们不是已校准的临床概率。

常用文件：

- `output/bulk/predictions.csv`：开发OOF与回顾性评分。
- `output/scRNA/replay_verification.json`：冻结评分、患者抽样及对照核对。
- `output/spatial/reproduced_patient_predictions.csv`：两张切片概率的患者均值。
- `output/paired/S13_fusion_vs_MMR4_macro.csv`：融合相对MMR4的配对增量。
- `output/stability/S14.csv`：17候选亚型的入选频率。
- `output/summary.json`：各分支完成状态。

示例数值可按约1e-10容差比较。抽样区间以已有特征和模型为条件；已查看队列仍为回顾性评价，空间代理迁移不证明与单细胞实测特征等价。

## 八、进一步文档

- [逐步使用教程](docs/TUTORIAL.md)
- [输入格式、随附文件及外部原始文件清单](docs/INPUT_FILES.md)
- [完整参数参考](docs/PARAMETERS.md)
- [常见错误与解释](docs/TROUBLESHOOTING.md)
- [第三方许可说明](THIRD_PARTY_NOTICES.md)

## License

项目原创代码及文档采用 **MIT License**，完整条款见 [LICENSE](LICENSE)。随附第三方软件保留原许可，原始数据条件仍适用。
