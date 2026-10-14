#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Apply locked sample-level scRNA QC and recompute GSE200997 metrics."""

# Packaging-only path resolution; archived originals retain exact source bytes.
from pathlib import Path as _MSIPackagePath
import sys as _MSIPackageSys
for _MSIPackageParent in _MSIPackagePath(__file__).resolve().parents:
    if (_MSIPackageParent / "project_paths.py").is_file():
        _MSIPackageSys.path.insert(0, str(_MSIPackageParent))
        break
from project_paths import project_root
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = project_root()
BASE = ROOT / '最终模型/data/scRNA/result/diagnostics/GSE200997_locked_shared18'
OUT = ROOT / '最终模型/data/scRNA/result/diagnostics/GSE200997_qc_total_cells_gt200'
OUT.mkdir(parents=True, exist_ok=True)
MIN_TOTAL_CELLS = 200

qc = pd.read_csv(BASE / 'sample_cell_QC.csv')
qc['qc_rule'] = f'total_cells >= {MIN_TOTAL_CELLS}'
qc['qc_pass'] = qc['total_cells'] >= MIN_TOTAL_CELLS
qc['exclusion_reason'] = np.where(qc['qc_pass'], '', f'total_cells < {MIN_TOTAL_CELLS}')
qc.to_csv(OUT / 'sample_cell_QC_decision.csv', index=False)

pred = pd.concat([pd.read_csv(BASE / 'local_predictions.csv'), pd.read_csv(BASE / 'peer_predictions.csv')], ignore_index=True)
keep = set(qc.loc[qc.qc_pass, 'sampleID'])
pred = pred[pred.sampleID.isin(keep)].copy()
pred.to_csv(OUT / 'all_predictions.csv', index=False)
labels = pd.read_csv(BASE / 'clinical_labels.csv').set_index('sampleID').loc[sorted(keep)]

matrix = pred.pivot(index='sampleID', columns='model', values='score').reindex(labels.index)
eligible = matrix.dropna().index
y = labels.loc[eligible, 'y'].to_numpy()
pos = np.flatnonzero(y == 1)
neg = np.flatnonzero(y == 0)
rng = np.random.default_rng(2026)
ip = rng.choice(len(pos), size=(2000, len(pos)), replace=True)
ineg = rng.choice(len(neg), size=(2000, len(neg)), replace=True)
rows = []
for name in matrix.columns:
    s = matrix.loc[eligible, name].to_numpy()
    auc = float(roc_auc_score(y, s))
    pair = (s[pos, None] > s[neg]).astype(float) + .5 * (s[pos, None] == s[neg])
    boot = pair[ip[:, :, None], ineg[:, None, :]].mean(axis=(1, 2))
    rows.append({'model': name, 'auc': auc, 'ci_low': float(np.quantile(boot, .025)),
                 'ci_high': float(np.quantile(boot, .975)), 'n': len(y),
                 'n_MSI': int(y.sum()), 'n_MSS': int((y == 0).sum()),
                 'epithelial_proxy': bool(pred.loc[pred.model.eq(name), 'epithelial_proxy'].iloc[0])})
summary = pd.DataFrame(rows).sort_values('auc', ascending=False)
summary.to_csv(OUT / 'AUROC_summary.csv', index=False)
ranking = pred[pred.model.eq('Shared18 + MLH1 (LDA)')].sort_values('score', ascending=False).copy()
ranking['rank'] = np.arange(1, len(ranking) + 1)
ranking.to_csv(OUT / 'shared18_patient_ranking.csv', index=False)

report = ['# GSE200997 样本级质量控制后的结果', '',
    '预先固定样本级 QC 条件为总细胞数 ≥200。该条件用于避免样本级细胞组成比例和 MLH1 均值由极少数细胞决定。', '',
          f'GSE200997 共 {len(qc)} 个肿瘤样本，{int(qc.qc_pass.sum())} 个通过 QC，{int((~qc.qc_pass).sum())} 个排除。T_cac5 仅有 {int(qc.loc[qc.sampleID.eq("T_cac5"), "total_cells"].iloc[0])} 个细胞，低于预设门槛，因样本级 QC 不达标排除。', '',
          '该排除规则在查看模型评分前确定，不依据 MSI 标签或模型分数。包含全部样本的历史敏感性结果保留在 GSE200997_locked_shared18 目录。', '',
          '| Model | AUROC | 95% CI | n | MSI/MSS |', '|---|---:|---|---:|---|']
for r in summary.itertuples():
    report.append(f'| {r.model} | {r.auc:.3f} | {r.ci_low:.3f}-{r.ci_high:.3f} | {r.n} | {r.n_MSI}/{r.n_MSS} |')
report += ['', '置信区间为2000次按结局分层的患者级bootstrap，随机种子2026；本轮仍不设置分类阈值，不报告错误率或BA。']
(OUT / '结果说明.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
(OUT / 'qc_manifest.json').write_text(json.dumps({'seed': 2026, 'min_total_cells_inclusive': MIN_TOTAL_CELLS,
    'rule': 'total_cells >= 200', 'n_input': int(len(qc)), 'n_pass': int(qc.qc_pass.sum()),
    'n_excluded': int((~qc.qc_pass).sum()), 'excluded_samples': qc.loc[~qc.qc_pass, 'sampleID'].tolist()}, ensure_ascii=False, indent=2), encoding='utf-8')
print(summary[['model', 'auc', 'ci_low', 'ci_high', 'n', 'n_MSI', 'n_MSS']].to_string(index=False))
