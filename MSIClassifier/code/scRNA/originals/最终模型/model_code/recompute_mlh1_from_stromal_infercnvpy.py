#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Recompute malignant-epithelial MLH1 using the stromal-reference CNV branch."""
from pathlib import Path
import json
import os
import joblib
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from sklearn.metrics import roc_auc_score

ROOT = Path('/home/wyw/rstudio/MSI')
FINAL = ROOT / '最终模型'
BRANCH = Path(os.environ.get(
    'INFERCNVPY_BRANCH_DIR',
    str(FINAL / 'data/scRNA/result/diagnostics/infercnvpy_stromal_only_seed2026'),
))
REFERENCE_QUANTILE = float(os.environ.get('INFERCNVPY_MLH1_REFERENCE_QUANTILE', '0.95'))
if not 0 < REFERENCE_QUANTILE < 1:
    raise ValueError('INFERCNVPY_MLH1_REFERENCE_QUANTILE must be between 0 and 1')
OUTPUT_LABEL = os.environ.get('INFERCNVPY_MLH1_OUTPUT_LABEL', 'stromal_only').strip()
OUTPUT_SUFFIX = ('_' + OUTPUT_LABEL) if OUTPUT_LABEL else ''
OUT = FINAL / ('data/scRNA/result/diagnostics/infercnvpy_stromal_mlh1_impact_q%02d%s_seed2026' % (round(REFERENCE_QUANTILE * 100), OUTPUT_SUFFIX))
OUT.mkdir(parents=True, exist_ok=True)

COHORTS = {
    'GSE178341': {
        'h5ad': ROOT / 'GSE178341.h5ad',
        'score': BRANCH / 'GSE178341/cell_malignancy_scores.csv.gz',
        'features': FINAL / 'data/scRNA/result/diagnostics/shared18_same_sample_benchmark_seed2026/GSE178341_benchmark_features.csv',
        'sample_col': 'sample',
        'class_col': 'cancer',
        'epithelial_mask': lambda o: o['clTopLevel'].astype(str).eq('Epi'),
    },
    'GSE132465': {
        'h5ad': ROOT / 'GSE132465.h5ad',
        'score': BRANCH / 'GSE132465/cell_malignancy_scores.csv.gz',
        'features': FINAL / 'data/scRNA/result/diagnostics/shared18_same_sample_benchmark_seed2026/GSE132465_benchmark_features.csv',
        'sample_col': 'Sample',
        'class_col': 'Class',
        'epithelial_mask': lambda o: o['Cell_type'].astype(str).eq('Epithelial cells'),
    },
}
RUN_COHORTS = set(filter(None, os.environ.get('INFERCNVPY_MLH1_DATASETS', '').split(',')))
COHORTS['GSE200997'] = {
    'h5ad': FINAL / 'data/scRNA/external/GSE200997/GSE200997.h5ad',
    'score': BRANCH / 'GSE200997/cell_malignancy_scores.csv.gz',
    'features': FINAL / 'data/scRNA/result/diagnostics/GSE200997_locked_shared18/benchmark_features.csv',
    'sample_col': 'Sample', 'class_col': 'Class',
    'migration': FINAL / 'data/scRNA/result/seed2026_migration_qc/GSE200997/process/GSE200997_tosica_migration_cells.csv',
}

def feature_table(cfg):
    table = pd.read_csv(cfg['features'])
    if 'migration' in cfg:
        # This migration file already applied the fixed total-cell QC rule.
        samples = pd.read_csv(cfg['migration'], usecols=['Sample'])['Sample'].unique()
        table = table[table.sampleID.isin(samples)].copy()
    return table

def column_values(adata, gene):
    names = pd.Index(adata.var_names.astype(str))
    loc = np.flatnonzero(names.to_numpy() == gene)
    if len(loc) != 1:
        raise ValueError(f'{adata.filename}: expected one {gene}, found {len(loc)}')
    x = adata[:, int(loc[0])].X
    if sparse.issparse(x):
        return np.asarray(x.toarray()).ravel().astype(float)
    return np.asarray(x).ravel().astype(float)

def read_cohort(name, cfg):
    scores = pd.read_csv(cfg['score'], index_col=0)
    scores.index = scores.index.astype(str)
    if not scores.index.is_unique:
        raise ValueError('Duplicate CNV cell IDs; rerun with unique cell selection')
    a = ad.read_h5ad(cfg['h5ad'], backed='r')
    a.obs_names_make_unique()
    obs = a.obs.copy()
    obs.index = obs.index.astype(str)
    obs['Sample'] = obs[cfg['sample_col']].astype(str)
    obs['Class_norm'] = obs[cfg['class_col']].astype(str).str.lower().map({'tumor': 'Tumor', 'normal': 'Normal'})
    if name == 'GSE132465':
        obs['Class_norm'] = obs[cfg['class_col']].astype(str)
    if 'migration' in cfg:
        migration = pd.read_csv(cfg['migration']).set_index('cell_id')
        assert migration.index.is_unique
        obs['is_epithelial'] = migration.major_prediction.reindex(obs.index).eq('Epi')
    else:
        obs['is_epithelial'] = cfg['epithelial_mask'](obs).to_numpy()
    obs['MLH1'] = column_values(a, 'MLH1')
    reference_mask = scores['cnv_reference'].notna() & scores['cnv_reference'].astype(str).ne('')
    threshold = float(scores.loc[reference_mask, 'cnv_score'].astype(float).quantile(REFERENCE_QUANTILE))
    (OUT / f'{name}_reference_threshold.json').write_text(json.dumps({
        'dataset': name,
        'reference_quantile': REFERENCE_QUANTILE,
        'reference_categories': sorted(scores.loc[reference_mask, 'cnv_reference'].astype(str).unique().tolist()),
        'threshold': threshold,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    assert scores.index.isin(obs.index).all()
    obs = obs.join(scores[['cnv_score', 'cnv_reference']], how='inner', validate='one_to_one')
    obs['candidate_above_reference_quantile'] = obs['cnv_score'].astype(float) >= (threshold - 1e-12)
    selected = obs[
        obs['Class_norm'].eq('Tumor')
        & obs['is_epithelial']
        & obs['candidate_above_reference_quantile']
        & obs['cnv_reference'].fillna('').eq('')
    ].copy()
    summary = selected.groupby('Sample').agg(
        n_malignant_epithelial=('MLH1', 'size'),
        mean_MLH1_epithelial=('MLH1', 'mean'),
    ).reset_index().rename(columns={'Sample': 'sampleID'})
    selected[['Sample', 'MLH1', 'cnv_score']].to_csv(OUT / f'{name}_selected_epithelial_cells.csv.gz')
    old = feature_table(cfg)
    old = old[['sampleID', 'MMR', 'mean_MLH1_epithelial']].copy()
    result = old.merge(summary, on='sampleID', how='left', suffixes=('_old', '_new'))
    result.insert(0, 'dataset', name)
    result['n_malignant_epithelial'] = result['n_malignant_epithelial'].fillna(0).astype(int)
    result['delta_MLH1_new_minus_old'] = result['mean_MLH1_epithelial_new'] - result['mean_MLH1_epithelial_old']
    result['abs_delta_MLH1'] = result['delta_MLH1_new_minus_old'].abs()
    result.to_csv(OUT / f'{name}_sample_MLH1_stromal_infercnvpy.csv', index=False)
    a.file.close()
    return result

def main():
    artifact = joblib.load(FINAL / 'model/reference/shared18_LDA_zscore2026/model.joblib')
    features = list(artifact['features'])
    rows = []
    predictions = []
    for name, cfg in COHORTS.items():
        if RUN_COHORTS and name not in RUN_COHORTS:
            continue
        x = read_cohort(name, cfg)
        old = feature_table(cfg)
        new = old.merge(x[['sampleID', 'mean_MLH1_epithelial_new']], on='sampleID', how='left')
        new['mean_MLH1_epithelial'] = new['mean_MLH1_epithelial_new']
        old_valid = old.dropna(subset=features).copy()
        new_valid = new.dropna(subset=features).copy()
        paired_old = old_valid[old_valid.sampleID.isin(new_valid.sampleID)]
        for label, table in [('original', old_valid), ('original_matched', paired_old), ('stromal_infercnvpy', new_valid)]:
            score = artifact['pipeline'].predict_proba(table[features].to_numpy())[:, 1] if len(table) else np.array([])
            y = table['MMR'].eq('MSI').astype(int).to_numpy()
            predictions.extend(dict(dataset=name, definition=label, sampleID=s, y=int(v), score=float(p)) for s,v,p in zip(table.sampleID,y,score))
            rows.append({
                'dataset': name, 'definition': label, 'n': len(table), 'MSI': int(y.sum()),
                'AUROC': float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else np.nan,
                'score_mean': float(score.mean()),
            })
        print(name)
        print(x[['sampleID', 'n_malignant_epithelial', 'mean_MLH1_epithelial_new', 'delta_MLH1_new_minus_old']].to_string(index=False))
    pd.DataFrame(rows).to_csv(OUT / 'locked_model_metrics.csv', index=False)
    pd.DataFrame(predictions).to_csv(OUT / 'sample_predictions.csv', index=False)
    all_changes = []
    for name in COHORTS:
        if RUN_COHORTS and name not in RUN_COHORTS:
            continue
        x = pd.read_csv(OUT / f'{name}_sample_MLH1_stromal_infercnvpy.csv')
        all_changes.append(x.assign(dataset=name))
    changes = pd.concat(all_changes, ignore_index=True)
    changes.to_csv(OUT / 'all_sample_MLH1_stromal_infercnvpy.csv', index=False)
    manifest = {
        'seed': 2026,
        'reference_branch': str(BRANCH),
        'reference_quantile': REFERENCE_QUANTILE,
        'definition': 'Class=Tumor, epithelial annotation, and infercnvpy cnv_score >= reference quantile',
        'model': str(FINAL / 'model/reference/shared18_LDA_zscore2026/model.joblib'),
        'interpretation': 'Sensitivity analysis; ecology features and locked model were unchanged. Samples with no inferred malignant epithelial cells have missing MLH1 and are excluded from score calculation.',
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False))

if __name__ == '__main__':
    main()
