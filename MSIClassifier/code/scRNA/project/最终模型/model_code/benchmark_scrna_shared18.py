"""Matched-sample AUROC benchmark without external-cohort model selection."""

# Packaging-only path resolution; archived originals retain exact source bytes.
from pathlib import Path as _MSIPackagePath
import sys as _MSIPackageSys
for _MSIPackageParent in _MSIPackagePath(__file__).resolve().parents:
    if (_MSIPackageParent / "project_paths.py").is_file():
        _MSIPackageSys.path.insert(0, str(_MSIPackageParent))
        break
from project_paths import project_root
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import run_scrna_metadata_adjusted_pipeline as core
import scrna_clinical_ablation as ab
from recompute_scrna_zscore_audits import load_data

OUT = core.FINAL / 'data/scRNA/result/diagnostics/shared18_same_sample_benchmark_seed2026'
REPO = core.FINAL / 'model/reference/peer_msisensor_rna_official'
GROUPS = ['B', 'Epi', 'Mast', 'Myeloid', 'Plasma', 'Strom', 'TNKILC']


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'cell_panels').mkdir(exist_ok=True)
    train, external, mt, me, _, folds = load_data()
    genes = json.loads((OUT / 'official_genes.json').read_text())
    panel = genes + [g for g in ['MLH1', 'MSH2', 'MSH6', 'PMS2'] if g not in genes]
    (OUT / 'extracted_genes.json').write_text(json.dumps(panel, indent=2))
    ledger = pd.concat([mt.assign(dataset='GSE178341'), me.assign(dataset='GSE132465')])
    fold_map = {train.iloc[i].sampleID: j for j, (_, test) in enumerate(folds, 1) for i in test}
    ledger['fold'] = ledger.sampleID.map(fold_map).fillna(0).astype(int)
    ledger.to_csv(OUT / 'sample_ledger.csv', index=False)
    for dataset, table in [('GSE178341', train), ('GSE132465', external)]:
        with h5py.File(core.FINAL.parent / (dataset + '.h5ad'), 'r') as handle:
            obs, var = ad.io.read_elem(handle['obs']), ad.io.read_elem(handle['var'])
            sample_col, tumor_col = ('sample', 'cancer') if dataset == 'GSE178341' else ('Sample', 'Class')
            sample = obs[sample_col].astype(str).to_numpy()
            keep = obs[tumor_col].astype(str).str.lower().eq('tumor').to_numpy() & np.isin(sample, table.sampleID)
            selected_obs = obs.loc[keep].copy()
            if dataset == 'GSE178341':
                major = selected_obs.majorCluster.astype(str)
            else:
                migrated = pd.read_csv(core.FINAL / 'data/scRNA/result/process/GSE132465_tosica_migration_cells.csv')
                assert migrated.cell_id.is_unique
                major = migrated.set_index('cell_id').major_prediction.reindex(selected_obs.index)
                assert major.notna().all()
            assert set(major) <= set(GROUPS)
            counts = pd.crosstab(selected_obs[sample_col], major).reindex(index=table.sampleID, columns=GROUPS, fill_value=0)
            assert counts.sum(axis=1).gt(0).all()
            counts.to_csv(OUT / (dataset + '_major_counts.csv'))
            counts.div(counts.sum(axis=1), axis=0).to_csv(OUT / (dataset + '_major_fractions.csv'))
            symbols = var.index.astype(str).to_numpy()
            locations = [np.flatnonzero(symbols == gene) for gene in panel]
            assert all(len(v) for v in locations)
            mapping = pd.DataFrame([dict(gene=g, index=int(i), gene_id=str(var.iloc[i].get('gene_ids', '')))
                                    for g, indices in zip(panel, locations) for i in indices])
            mapping.to_csv(OUT / (dataset + '_gene_mapping.csv'), index=False)
            col_map = sparse.csr_matrix((np.ones(sum(map(len, locations))),
                (np.concatenate(locations), np.repeat(np.arange(len(panel)), [len(v) for v in locations]))),
                shape=(len(symbols), len(panel)))
            group = handle['layers/counts']
            is_sparse = isinstance(group, h5py.Group)
            indptr = group['indptr'][:] if is_sparse else None
            cells, libraries, cell_names = [], [], []
            for start in range(0, len(obs), 512):
                stop = min(start + 512, len(obs))
                select = keep[start:stop]
                if not select.any():
                    continue
                if is_sparse:
                    a, b = indptr[start], indptr[stop]
                    block = sparse.csr_matrix((group['data'][a:b], group['indices'][a:b],
                        indptr[start:stop+1] - a), shape=(stop-start, len(symbols)))[select]
                    assert np.isfinite(block.data).all() and (block.data >= 0).all()
                    small = (block @ col_map).toarray()
                else:
                    block = group[start:stop, :][select]
                    assert np.isfinite(block).all() and (block >= 0).all()
                    small = np.asarray(block @ col_map)
                cells.append(small.astype(np.float32))
                libraries.append(np.asarray(block.sum(axis=1)).ravel().astype(float))
                cell_names.append(sample[start:stop][select])
            x, lib, ids = np.concatenate(cells), np.concatenate(libraries), np.concatenate(cell_names)
            assert len(x) == keep.sum() and (lib > 0).all()
            pseudobulk = []
            for sid in table.sampleID:
                mask = ids == sid
                assert mask.any()
                np.savez_compressed(OUT / 'cell_panels' / (dataset + '_' + sid + '.npz'), counts=x[mask], library_size=lib[mask])
                pseudobulk.append(np.log2(1 + x[mask].sum(axis=0, dtype=float) / lib[mask].sum() * 1e6))
            pd.DataFrame(pseudobulk, index=table.sampleID, columns=panel).to_csv(OUT / (dataset + '_pseudobulk_log2CPM.csv'))
            print(dataset, len(x), 'tumor cells extracted; duplicated panel symbols', int(sum(len(v)>1 for v in locations)), flush=True)
    write_manifest()


def write_manifest():
    manifest = {'seed': 2026, 'model': 'fixed18 Z-score shrinkage LDA + epithelial MLH1',
        'repo_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=REPO, text=True).strip(),
        'official_model_sha256': hashlib.sha256((REPO / 'model/TCGA.CRC.model.pkl').read_bytes()).hexdigest(),
        'paper': 'https://doi.org/10.1093/gpbjnl/qzae004',
        'protocol': 'All comparators prespecified; no external tuning; same 64/23 samples and patient folds.',
        'pseudobulk': 'Sum raw counts by sample, sum duplicate gene symbols, CPM using all-gene library, log2(1+CPM).',
        'cell_mean': 'Tumor cells; >=20% of397 panel genes detected, else top20. Raw zeros imputed with per-gene mean over all tumor cells of that sample before log2(1+x). Official gene-wise-within-cell standardization ddof1, then score cells and average. Log-space imputation is a separate sensitivity.',
        'limitations': 'Fixed18 selected historically, not a nested selection estimate. GSE132465 repeatedly examined and retrospective. Official paper scRNA preprocessing has interpretive details, so cell mean is a paper-informed reconstruction, not an exact replication. No new independent validation.'}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))


def local_models():
    train, external, mt, me, _, folds = load_data()
    contract = pd.read_csv(core.CLASS_CONTRACT_PATH).set_index('class_name').group_class
    groups = sorted(set(contract.loc[list(core.CURRENT_SHARED18)]))
    tables = []
    for dataset, data in [('GSE178341', train), ('GSE132465', external)]:
        table = data.copy()
        coarse = pd.read_csv(OUT / (dataset + '_major_fractions.csv'), index_col=0).reindex(data.sampleID)
        expr = pd.read_csv(OUT / (dataset + '_pseudobulk_log2CPM.csv'), index_col=0).reindex(data.sampleID)
        for group in GROUPS:
            table['major_' + group] = coarse[group].to_numpy()
        for group in groups:
            members = [f for f in core.CURRENT_SHARED18 if contract[f] == group]
            table['collapsed_' + group] = table[members].sum(axis=1)
        table = pd.concat([table.reset_index(drop=True), expr.add_prefix('gene_').reset_index(drop=True)], axis=1)
        table.to_csv(OUT / (dataset + '_benchmark_features.csv'), index=False)
        tables.append(table)
    lda = core.make_models(2026)['lda_shrinkage']
    logistic = LogisticRegression(C=1.0, penalty='l2', max_iter=10000, random_state=2026)
    mlh1 = [core.MLH1_FEATURE]
    genes = json.loads((OUT / 'official_genes.json').read_text())
    configurations = {
        'Shared18 + MLH1 (LDA)': (list(core.CURRENT_SHARED18) + mlh1, lda),
        'Shared18 only (LDA)': (list(core.CURRENT_SHARED18), lda),
        'MLH1 only (LDA)': (mlh1, lda),
        'Major7 + MLH1 (LDA)': (['major_'+g for g in GROUPS] + mlh1, lda),
        'Major7 + MLH1 (Logistic)': (['major_'+g for g in GROUPS] + mlh1, logistic),
        'Collapsed18 + MLH1 (LDA)': (['collapsed_'+g for g in groups] + mlh1, lda),
        'MMR4 expression (Logistic)': (['gene_'+g for g in ['MLH1','MSH2','MSH6','PMS2']], logistic),
        '397-gene expression (Logistic)': (['gene_'+g for g in genes], logistic)}
    scores, metrics = [], []
    for name, (features, model) in configurations.items():
        met, pred = core.evaluate_unadjusted_candidate(*tables, mt, me, features, model, name, folds, 2026, normalization='zscore')
        scores.append(pred.assign(model=name)); metrics.append(met)
        print(name, met[['split','auc']].to_dict('records'), flush=True)
    # A stronger expression comparator selects regularization inside each outer fold.
    features = ['gene_'+g for g in genes]
    selections = []
    tuned_parts = []
    for fold, (fit, test) in enumerate(folds, 1):
        inner = core.make_patient_grouped_folds(mt.iloc[fit].y, mt.iloc[fit].patientID, n_splits=4, random_state=2026)
        search = GridSearchCV(Pipeline([('scale', StandardScaler()), ('model', logistic)]),
            {'model__C': [.001, .01, .1, 1., 10., 100.]}, cv=inner, scoring='roc_auc', n_jobs=1)
        search.fit(tables[0].iloc[fit][features], mt.iloc[fit].y)
        selections.append(dict(fold=fold, C=search.best_params_['model__C'], inner_auc=search.best_score_))
        part = pd.DataFrame({'sampleID':train.iloc[test].sampleID.to_numpy(), 'patientID':mt.iloc[test].patientID.to_numpy(),
            'y':mt.iloc[test].y.to_numpy(), 'score':search.predict_proba(tables[0].iloc[test][features])[:,1],
            'split':'train_grouped_oof', 'dataset':'GSE178341', 'fold':fold})
        tuned_parts.append(part)
    inner = core.make_patient_grouped_folds(mt.y, mt.patientID, n_splits=5, random_state=2026)
    search = GridSearchCV(Pipeline([('scale', StandardScaler()), ('model', logistic)]),
        {'model__C': [.001, .01, .1, 1., 10., 100.]}, cv=inner, scoring='roc_auc', n_jobs=1)
    search.fit(tables[0][features], mt.y)
    selections.append(dict(fold=0, C=search.best_params_['model__C'], inner_auc=search.best_score_))
    for table, meta, split, dataset in [(tables[0],mt,'train_full_fit','GSE178341'), (tables[1],me,'external','GSE132465')]:
        tuned_parts.append(pd.DataFrame({'sampleID':table.sampleID, 'patientID':meta.patientID.to_numpy(),
            'y':meta.y.to_numpy(), 'score':search.predict_proba(table[features])[:,1], 'split':split, 'dataset':dataset, 'fold':0}))
    scores.append(pd.concat(tuned_parts).assign(model='397-gene expression (tuned Logistic)'))
    pd.DataFrame(selections).to_csv(OUT / 'expression_regularization_selection.csv', index=False)
    pd.concat(scores).to_csv(OUT / 'local_predictions.csv', index=False)
    pd.concat(metrics).to_csv(OUT / 'local_metrics.csv', index=False)
    (OUT / 'local_model_specifications.json').write_text(json.dumps({name: {'features':f, 'parameters':m.get_params()} for name,(f,m) in configurations.items()}, indent=2, default=str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare','local','manifest'])
    args = parser.parse_args()
    {'prepare':prepare, 'local':local_models, 'manifest':write_manifest}[args.stage]()
