"""Small, evidence-backed edits to the user's newly uploaded manuscript."""
from pathlib import Path
import json
import pandas as pd

HERE = Path(__file__).resolve().parent
items = json.loads((HERE / 'source_inventory.json').read_text())
old = {x['block']: x['text'] for x in items if 'text' in x}
changes = {}
def replace(i, before, after):
    value = changes.get(i, old[i])
    assert value.count(before) == 1, (i, before)
    changes[i] = value.replace(before, after)
def append(i, value): changes[i] = changes.get(i, old[i]) + value

replace(20, '逐队列结果与各轮MMR4训练阈值分别见表2B、表2C及补充表S12–S13。',
            '宏平均结果见表2B，逐队列结果见图4B及补充表S13，各轮MMR4阈值见表2C及补充表S12。')
replace(35, '在本研究开发数据的既定比较中，加入MSH2、MSH6和PMS2未提高所用判别指标。结合MLH1失活的生物学依据[33]，分类模型保留MLH1，四基因表达分布用于描述性比较。',
            '结合MLH1失活的生物学证据[33]及既往模型配置，分类模型的MMR表达组分保留MLH1；四基因表达分布用于描述性比较。')
replace(38, '在通过样本QC的肿瘤细胞中', '从通过样本QC的肿瘤组织样本所含的全部细胞中')
append(43, '通过cell2location的export_posterior取得200次抽样的参数后验均值（post_sample_means），细胞丰度采用means_cell_abundance_w_sf。结合固定参考表达谱，按细胞类型条件期望计数分配公式，将各spot的观测UMI计数分配至CNV候选恶性上皮成分；公式及后验输出名称见补充方法。')
replace(45, '丰度之和。该代理特征',
            '丰度之和。将候选成分在12,079个共享且经参考过滤的模型基因上的分配计数按spot归一化至总量10,000，再进行自然对数log(1+x)转换，所得MLH1值定义为ReconstructedMLH1_i。该重建使用参数后验均值代入公式计算，归一化分母为候选成分在全部模型基因上的分配计数总和。该代理特征')
replace(48, '分析保留连续的线性尺度表达，并记录多项分布采样中的四舍五入操作。',
            '连续线性表达矩阵予以存档；运行前对整张bulk输入逐项使用NumPy rint取最近整数（半整数取最近偶数）并转换为int64，取整矩阵用于初始Gibbs推断、队列内参考更新及最终丰度估计。')
replace(50, '分别使用Python BayesPrism。参考固定为GSE178341的71个非上皮亚型及Epi。',
            '分别采用BayesPrism[32]的Python移植pybayesprism 0.1.0[37]（含存档本地修复；仓库ziluwang829/pyBayesPrism）。参考由GSE178341.h5ad的原始counts层构建，固定为71个非上皮亚型及Epi，不进行逐细胞总量归一化或对数转换。')
replace(50, 'Gibbs采样运行1,000次迭代，', 'Gibbs链长度参数设为1,000，')
replace(50, '输出丰度按所属主要细胞大类归一化。输入、基因覆盖和参数记录见随附复现资料。',
            '采用更新参考后的后验平均丰度posterior_theta_f.theta（final_fractions.csv），并以各亚型丰度除以同一样本所属主要细胞大类的丰度总和，形成分类器输入。各队列独立更新参考，软件修复、输入层、实际运行基因及采样记录见补充方法和随附复现资料。')
replace(52, 'MMR4使用MLH1、MSH2、MSH6和PMS2的队列内标准化表达。',
            'MMR4使用MLH1、MSH2、MSH6和PMS2的队列内标准化表达。标准化前的四基因值取自各队列存档基因表达矩阵（expression.csv）：TCGA-COADREAD为log₂(RSEM标准化表达+1)，七个GEO队列为处理后的log₂微阵列表达。MMR4直接使用上述对数尺度值，不复用BayesPrism的线性尺度反变换或取整结果；矩阵对应关系及数值核对见补充表S12。')
replace(54, '及MLH1、MSH2、MSH6、PMS2的阈值字典序确定，前两个排序指标均保留12位小数。',
            '及MLH1、MSH2、MSH6、PMS2阈值向量的字典序最小值确定。训练宏平均AUROC与BA在排序比较前均舍入至小数点后12位，该舍入用于并列判定。')
replace(59, '交叉验证以患者为分组单位，bootstrap以患者为聚类单位，同一患者的重复样本共同抽取。',
            'AUROC直接以64个样本的预测分数计算，不先进行患者内汇总；交叉验证以患者为分组单位，置信区间以患者为聚类单位估计，同一患者的重复样本共同抽取。')
replace(66, 'AUROC的置信区间与7亚型和12亚型融合模型的配对AUROC差值采用2,000次队列内结局分层样本bootstrap估计，固定预测分数及模型，不纳入重新拟合或特征筛选的不确定性。',
            'AUROC的置信区间、7亚型与12亚型融合模型的配对差值，以及12亚型融合相对MMR4单分支的配对差值，均采用2,000次队列内结局分层样本bootstrap估计。融合与MMR4按同一队列、同一样本及对应MMR4留出轮次配对，每次使用相同抽样索引；宏平均差值由同次各队列差值等权汇总，以第2.5和97.5百分位构成95%置信区间。分析固定预测分数、队列标准化参数及各轮阈值，不重新拟合模型或筛选特征；抽样索引及单次初始化并推进的2026随机流记录随附提供。')
append(78, '表2B的性能指标保留4位小数，增量及其置信区间由未舍入的原值计算，显示值相减可能产生末位差异。融合相对MMR4的逐队列和宏平均配对区间见补充表S13。')
replace(87, '根据开发阶段的增量比较和MLH1的生物学依据，', '结合既往模型配置和MLH1的生物学证据，')
append(90, '候选恶性上皮比例单独输入时，三个队列的AUROC分别为0.861、0.967和0.538；联合18亚型后分别为0.967、0.974和1.000。将MLH1均值与比例分开输入并联合18亚型时，AUROC分别为0.957、0.974和1.000，而采用均值与比例乘积的加权联合模型分别为0.978、0.961和1.000（补充表S7），两种特征形式的表现具有队列差异。')
macro = pd.read_csv(HERE / 'paired_increment/S13_fusion_vs_MMR4_macro.csv').set_index('scope')
d, r = macro.loc['development_OOF'], macro.loc['retrospective']
replace(119, '相较MMR4单分支，宏平均AUROC点估计分别增加0.0315和0.0335。',
             f'相较MMR4单分支，宏平均AUROC的配对增量分别为{d.Delta_AUROC_Fusion_minus_MMR4:.4f}（95% CI {d.CI_low:.4f}–{d.CI_high:.4f}）和{r.Delta_AUROC_Fusion_minus_MMR4:.4f}（{r.CI_low:.4f}–{r.CI_high:.4f}）（补充表S13）。')
replace(119, '融合相对MMR4的宏平均增量在此为点估计，不能据此认定每个队列均获益。',
             '配对区间以固定预测分数及各轮阈值为条件；逐队列增量具有异质性，GSE24550和GSE25071的点估计为负，部分队列的区间包含0。')
replace(119, '队列分组沿用亚型模型的开发用途；配对区间以固定预测分数及各轮阈值为条件；逐队列增量',
             '队列分组沿用亚型模型的开发用途。配对区间以固定预测分数及各轮阈值为条件，逐队列增量')
replace(120, '该区间对应7亚型与12亚型配置之间的差异，不是融合模型相对MMR4单分支的增量区间。',
             '配置间比较与融合相对MMR4的分支增量分别列于补充表S13。')
append(139, '其他MMR基因在当前加权联合模型中的独立增量仍需在相同预测框架下评价。')
replace(144, 'S12记录MMR4留一队列阈值，S13记录12亚型MLP、MMR4和融合的逐队列结果、评分及7亚型和12亚型的配对比较。',
             'S12记录MMR4留一队列阈值及标准化前的输入尺度，S13记录12亚型MLP、MMR4和融合的逐队列结果、评分、7亚型与12亚型的配对比较，以及融合相对MMR4的逐队列和宏平均配对增量区间。补充方法列示BayesPrism实现及丰度输出、空间表达重建公式、后验汇总和归一化基因范围。')
for i, value in old.items():
    value = changes.get(i, value)
    if 'bulk12' in value: changes[i] = value.replace('bulk12', 'bulk 12')

changes[183] = '32. Chu T, Wang Z, Pe’er D, Danko CG. Cell type and gene expression deconvolution with BayesPrism enables Bayesian integrative analysis across bulk and single-cell RNA sequencing in oncology. Nat Cancer. 2022;3(4):505-517. doi:10.1038/s43018-022-00356-3. PMID:35469013.'
new_reference = '37. Wang Z. pyBayesPrism: Python implementation of BayesPrism. Version 0.1.0 [Internet]. 2024 [cited 2026 Oct 14]. Available from: https://github.com/ziluwang829/pyBayesPrism.'
(HERE / 'text_changes.json').write_text(json.dumps({'paragraphs': changes, 'new_reference': new_reference}, ensure_ascii=False, indent=2))
print('Prepared local changes in', len(changes), 'paragraphs and one added software reference')
