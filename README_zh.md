# MSIClassifier 使用说明

MSIClassifier从bulk、单细胞和空间基因表达数据生成结直肠癌MSI/dMMR研究评分。程序构建细胞亚型及MMR相关特征，再应用随附分类器。运行不需要MSI标签或BAM文件。

[English README](README.md)

## 安装

已测试环境为Linux、Python 3.12。以下命令均在仓库目录中执行。

```bash
conda create -n msiclassifier python=3.12 pip
conda activate msiclassifier
cd MSIClassifier
python -m pip install -r requirements.txt
```

单细胞和空间表达分析另需安装：

```bash
python -m pip install -r requirements-expression.txt
```

默认使用CPU；有兼容GPU时可添加 `--device cuda:0`。[详细安装说明](docs/EXPRESSION_WORKFLOW.md)。

## 下载参考资源

从仓库的 **Releases** 下载所需文件：

| Analysis | Required reference archives |
| --- | --- |
| Bulk expression | Included in the code package |
| Single-cell expression | `MSIClassifier_RawReference_part1.zip` + `MSIClassifier_RawReference_part2.zip` |
| Spatial expression | `MSIClassifier_RawReference_part2.zip` |

将所需资源包**解压到MSIClassifier目录内**，确认 `resources/raw_reference/manifest.json` 存在。

## 准备输入

| Analysis | Input |
| --- | --- |
| Bulk | Gene-by-sample CSV/TSV; first column: gene symbols |
| Single-cell | h5ad, 10x HDF5/MTX, loom or cell-by-gene CSV/TSV; specimen/tissue metadata |
| Spatial | Raw spot counts from one section; same supported matrix formats |

bulk需按实际尺度选择 `--scale linear`、`log2` 或 `log2p1`。芯片原始扫描文件先完成平台预处理；处理后的探针矩阵另需 `--probe-map`。

单细胞需在 `adata.obs` 或 `--metadata` CSV中提供 `Sample` 和 `Class`，其中 `Class=Tumor` 表示肿瘤组织。原始计数使用 `--scale counts`，已归一化表达使用 `--scale log1p-cp10k`。空间输入必须为原始计数。

[输入格式与示例](docs/EXPRESSION_WORKFLOW.md) · [完整参数说明](docs/EXPRESSION_PARAMETERS.md)

## 运行预测

```bash
# bulk：此示例假设输入为log2(线性表达+1)
python predict.py bulk --expression bulk_expression.csv --scale log2p1 \
  --dataset MY_BULK --output-dir output/my_bulk

# 单细胞：从h5ad的counts层读取原始计数
python predict.py scrna --expression cells.h5ad --layer counts --scale counts \
  --dataset MY_SCRNA --output-dir output/my_scRNA

# 空间：读取一张切片的原始spot计数
python predict.py spatial --expression section.h5ad --layer counts \
  --sample-id sectionA --output-dir output/sectionA
```

原始计数保存在 `X` 时，去掉 `--layer counts`。请使用新的输出目录；添加 `--check-only` 可先检查输入。[随附的可运行示例](examples/expression/README.md)。

## 输出结果

每次运行生成 `features.csv`、`scores.csv` 和 `run_manifest.json`。评分越高，模型越倾向MSI/dMMR；评分用于研究，不是经过校准的临床概率。空间评分基于重建的代理特征。

新的bulk队列默认输出亚型分支评分。可选的MMR4固定0.5∶0.5融合需使用匹配队列参数，见[融合使用说明](docs/EXPRESSION_WORKFLOW.md#optional-matching-cohort-mmr4-fusion)。

## 详细文档

- [完整流程与输入文件](docs/EXPRESSION_WORKFLOW.md)
- [参数说明](docs/EXPRESSION_PARAMETERS.md)
- [常见问题](docs/TROUBLESHOOTING.md)
- [已有特征评分及高级用法](docs/TUTORIAL.md)
- [论文分析复现](docs/REPRODUCING_ANALYSES.md)

## License

项目采用[MIT许可](LICENSE)。第三方组件保留其原有许可，见[第三方声明](THIRD_PARTY_NOTICES.md)。
