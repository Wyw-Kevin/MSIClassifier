#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run an infercnvpy CNV branch with normal epithelial/stromal references."""
from pathlib import Path
import json
import re
import sys
import os
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import infercnvpy as cnv

ROOT = Path('/home/wyw/rstudio/MSI')
REFERENCE_TYPE = os.environ.get('INFERCNVPY_REFERENCE_TYPE', 'stromal')
REFERENCE_LABEL = 'endothelial' if REFERENCE_TYPE == 'endothelial' else 'stromal'
REFERENCE_VARIANT = os.environ.get('INFERCNVPY_REFERENCE_VARIANT', 'default')
REFERENCE_MODE = os.environ.get('INFERCNVPY_REFERENCE_MODE', 'stromal_only')
OUT_NAME = 'infercnvpy_normal_epi_%s_seed2026' % REFERENCE_LABEL
if REFERENCE_VARIANT == 'all_normal_epithelial':
    OUT_NAME = 'infercnvpy_normal_epi_%s_all_normal_epi_seed2026' % REFERENCE_LABEL
if REFERENCE_MODE in {'stromal_only', 'normal_epithelial_only'}:
    OUT_NAME = 'infercnvpy_%s_seed2026' % REFERENCE_MODE
OUT = ROOT / ('最终模型/data/scRNA/result/diagnostics/' + OUT_NAME)
OUT = Path(os.environ.get('INFERCNVPY_OUTPUT_DIR', str(OUT)))
OUT.mkdir(parents=True, exist_ok=True)
SEED = 2026
MIN_TOTAL_CELLS = 200
MAX_TUMOR = 100000
MAX_REFERENCE = 20000
GENE_FILE = ROOT / 'inferCNV/GSE132465/geneFile.txt'

DATASETS = {
    'GSE178341': ROOT / 'GSE178341.h5ad',
    'GSE132465': ROOT / 'GSE132465.h5ad',
    'GSE200997': ROOT / '最终模型/data/scRNA/external/GSE200997/GSE200997.h5ad',
}

def load_positions(var_names):
    pos = pd.read_csv(GENE_FILE, sep='\t', header=None, names=['gene', 'chromosome', 'start', 'end'])
    pos = pos.drop_duplicates('gene').set_index('gene')
    keys = pd.Index(var_names.astype(str)).str.replace(r'-\d+$', '', regex=True)
    table = pd.DataFrame({'gene_key': keys, 'original': var_names.astype(str)})
    table = table[table.gene_key.isin(pos.index)].drop_duplicates('gene_key')
    table = table.join(pos, on='gene_key')
    return table

def sample_filter(obs):
    counts = obs.groupby('Sample', sort=False).size()
    keep = set(counts[counts >= MIN_TOTAL_CELLS].index.astype(str))
    return obs['Sample'].astype(str).isin(keep).to_numpy(), counts.to_dict()

def reference_labels(dataset, obs, cell_ids):
    ref = pd.Series('', index=obs.index, dtype='object')
    if dataset == 'GSE178341':
        # Use the independent normal-sample and broad epithelial annotations.
        # clMidwayPr=Epi_Normal is an R-inferCNV-derived label and is excluded.
        if REFERENCE_MODE != 'stromal_only':
            ref.loc[obs['cancer'].astype(str).eq('normal') & obs['clTopLevel'].astype(str).eq('Epi')] = 'normal_epithelial'
        if REFERENCE_TYPE == 'endothelial' and REFERENCE_MODE != 'normal_epithelial_only':
            ref.loc[obs['clMidwayPr'].astype(str).eq('Endo')] = 'endothelial'
        elif REFERENCE_MODE != 'normal_epithelial_only':
            ref.loc[obs['clTopLevel'].astype(str).eq('Strom')] = 'stromal'
    elif dataset == 'GSE132465':
        ref.loc[obs['Class'].astype(str).eq('Normal') & obs['Cell_type'].astype(str).eq('Epithelial cells')] = 'normal_epithelial'
        if REFERENCE_TYPE == 'stromal':
            migration = ROOT / '最终模型/data/scRNA/result/seed2026_migration/GSE132465/process/GSE132465_tosica_migration_cells.csv'
            m = pd.read_csv(migration, usecols=['cell_id', 'major_prediction']).set_index('cell_id')
            stromal = m.reindex(pd.Index(obs.index.astype(str)))['major_prediction'].astype(str).eq('Strom')
            ref.loc[stromal.to_numpy()] = 'stromal'
    elif dataset == 'GSE200997':
        migration = ROOT / '最终模型/data/scRNA/result/seed2026_migration_qc/GSE200997/process/GSE200997_tosica_migration_cells.csv'
        if not migration.exists():
            raise FileNotFoundError('Run seed-2026 TOSICA migration first: %s' % migration)
        m = pd.read_csv(migration, usecols=['cell_id', 'Class', 'major_prediction']).set_index('cell_id')
        ids = pd.Index(obs.index.astype(str))
        normal_epi = m.reindex(ids)['Class'].astype(str).eq('Normal') & m.reindex(ids)['major_prediction'].astype(str).eq('Epi')
        ref.loc[normal_epi.to_numpy()] = 'normal_epithelial'
        if REFERENCE_TYPE == 'stromal':
            stromal = m.reindex(ids)['major_prediction'].astype(str).eq('Strom')
            ref.loc[stromal.to_numpy()] = 'stromal'
    if REFERENCE_MODE == 'stromal_only':
        ref.loc[ref.ne('stromal')] = ''
    elif REFERENCE_MODE == 'normal_epithelial_only':
        ref.loc[ref.ne('normal_epithelial')] = ''
    return ref

def run_one(dataset, path):
    rng = np.random.default_rng(SEED)
    a = ad.read_h5ad(path)
    a.obs_names_make_unique()
    a.var_names = pd.Index(a.var_names.astype(str))
    a.var_names_make_unique()
    obs = a.obs.copy()
    if 'Sample' not in obs.columns:
        if 'samples' in obs.columns:
            obs['Sample'] = obs['samples'].astype(str)
        elif 'sample' in obs.columns:
            obs['Sample'] = obs['sample'].astype(str)
        else:
            raise ValueError('%s has no sample column' % dataset)
    if 'Class' not in obs.columns:
        if 'Condition' in obs.columns:
            obs['Class'] = obs['Condition'].astype(str).map({'Tumor': 'Tumor', 'Normal': 'Normal'})
        elif 'cancer' in obs.columns:
            obs['Class'] = obs['cancer'].astype(str).map({'tumor': 'Tumor', 'normal': 'Normal'})
        else:
            raise ValueError('%s has no Class/Condition column' % dataset)
    a.obs = obs
    keep, counts = sample_filter(obs)
    a = a[keep].copy()
    ref = reference_labels(dataset, a.obs, a.obs_names)
    a.obs['cnv_reference'] = ref.to_numpy()
    tumor = a.obs['Class'].astype(str).eq('Tumor').to_numpy()
    ref_mask = a.obs['cnv_reference'].ne('').to_numpy()
    tumor_ids = np.flatnonzero(tumor)
    ref_ids = np.flatnonzero(ref_mask)
    if len(tumor_ids) > MAX_TUMOR:
        tumor_ids = rng.choice(tumor_ids, MAX_TUMOR, replace=False)
    if len(ref_ids) > MAX_REFERENCE:
        ref_ids = rng.choice(ref_ids, MAX_REFERENCE, replace=False)
    selected = np.union1d(tumor_ids, ref_ids)
    a = a[selected].copy()
    assert a.obs_names.is_unique
    positions = load_positions(pd.Index(a.var_names))
    keep_genes = np.asarray(a.var_names.isin(positions['original']))
    a = a[:, keep_genes].copy()
    positions = load_positions(pd.Index(a.var_names)).set_index('original').loc[pd.Index(a.var_names)]
    a.var['chromosome'] = positions['chromosome'].to_numpy()
    a.var['start'] = positions['start'].to_numpy()
    a.var['end'] = positions['end'].to_numpy()
    observed_reference_categories = set(a.obs['cnv_reference'].astype(str))
    # Prefer both references when available; if a cohort has no normal sample,
    # retain the stromal reference alone rather than fabricating normal cells.
    if 'normal_epithelial' in observed_reference_categories:
        reference_categories = ['normal_epithelial']
        if REFERENCE_LABEL in observed_reference_categories:
            reference_categories.append(REFERENCE_LABEL)
        reference_selection = 'normal_epithelial_plus_%s' % REFERENCE_LABEL
    elif REFERENCE_LABEL in observed_reference_categories:
        reference_categories = [REFERENCE_LABEL]
        reference_selection = '%s_only_no_normal_sample' % REFERENCE_LABEL
    else:
        reference_categories = []
        reference_selection = 'none'
    if REFERENCE_MODE == 'stromal_only':
        reference_categories = [REFERENCE_LABEL] if REFERENCE_LABEL in observed_reference_categories else []
        reference_selection = '%s_only' % REFERENCE_LABEL
    elif REFERENCE_MODE == 'normal_epithelial_only':
        reference_categories = ['normal_epithelial'] if 'normal_epithelial' in observed_reference_categories else []
        reference_selection = 'normal_epithelial_only'
    if not reference_categories:
        raise ValueError('%s has no usable normal epithelial/%s reference cells' % (dataset, REFERENCE_LABEL))
    # infercnvpy expects log-scale expression; the archived h5ad X matrix is already log-normalized.
    cnv.tl.infercnv(a, reference_key='cnv_reference', reference_cat=reference_categories,
                    lfc_clip=3, window_size=100, step=10, dynamic_threshold=1.5,
                    exclude_chromosomes=('chrX', 'chrY'), chunksize=5000, n_jobs=4,
                    key_added='cnv', calculate_gene_values=False)
    cnv.tl.pca(a, use_rep='cnv', key_added='cnv_pca', random_state=SEED)
    cnv.pp.neighbors(a, use_rep='cnv_pca', key_added='cnv_neighbors', random_state=SEED)
    cnv.tl.leiden(a, neighbors_key='cnv_neighbors', key_added='cnv_leiden', random_state=SEED)
    cnv.tl.cnv_score(a, groupby='cnv_leiden', use_rep='cnv', key_added='cnv_score')
    ref_scores = a.obs.loc[a.obs['cnv_reference'].ne(''), 'cnv_score'].astype(float)
    threshold = float(ref_scores.quantile(.99)) if len(ref_scores) else np.nan
    # Use a tolerance because q99 and CSV/HDF5 round-tripping can differ by
    # machine epsilon; strict `>` can otherwise erase the boundary cluster.
    a.obs['malignant_infercnvpy_q99'] = a.obs['cnv_score'].astype(float) >= (threshold - 1e-12)
    a.obs['dataset'] = dataset
    out = OUT / dataset
    out.mkdir(exist_ok=True)
    a.obs[['dataset', 'Sample', 'Class', 'cnv_reference', 'cnv_leiden', 'cnv_score', 'malignant_infercnvpy_q99']].to_csv(out / 'cell_malignancy_scores.csv.gz', index=True, compression='gzip')
    summary = a.obs.groupby(['Class', 'cnv_reference'], dropna=False).size().reset_index(name='n_cells')
    summary.to_csv(out / 'reference_and_cell_counts.csv', index=False)
    sample = a.obs[a.obs['Class'].eq('Tumor')].groupby('Sample').agg(n_cells=('Class', 'size'), n_malignant=('malignant_infercnvpy_q99', 'sum'), mean_cnv_score=('cnv_score', 'mean')).reset_index()
    sample.to_csv(out / 'sample_malignancy_summary.csv', index=False)
    a.write(out / 'infercnvpy_result.h5ad', compression='gzip')
    manifest = {'dataset': dataset, 'seed': SEED, 'reference_variant': REFERENCE_VARIANT, 'input': str(path), 'gene_file': str(GENE_FILE),
        'unique_cell_selection': True, 'pca_random_state': SEED, 'neighbors_random_state': SEED,
        'min_total_cells_inclusive': MIN_TOTAL_CELLS, 'n_input_cells': int(len(obs)), 'n_after_sample_qc': int(keep.sum()),
        'n_cells_in_analysis': int(a.n_obs), 'n_genes_with_positions': int(a.n_vars),
        'reference_mode': REFERENCE_MODE,
        'reference_categories_requested': reference_categories,
        'reference_categories_observed': sorted(reference_categories),
        'reference_selection': reference_selection,
        'reference_cell_counts': a.obs['cnv_reference'].value_counts().to_dict(),
        'n_tumor_cells_analyzed': int(a.obs.Class.eq('Tumor').sum()), 'reference_q99_cnv_score': threshold,
        'parameters': {'lfc_clip': 3, 'window_size': 100, 'step': 10, 'dynamic_threshold': 1.5, 'exclude_chromosomes': ['chrX', 'chrY'], 'n_jobs': 4},
        'reference_threshold_tolerance': 1e-12,
        'interpretation': 'infercnvpy CNV score and q99 reference threshold are exploratory malignancy evidence; infercnvpy does not prescribe this q99 threshold.'}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(dataset, json.dumps(manifest, ensure_ascii=False))

only = set(filter(None, __import__('os').environ.get('INFERCNVPY_DATASETS', '').split(',')))
for name, path in DATASETS.items():
    if only and name not in only:
        continue
    run_one(name, path)
