"""Fixed shared18 LDA comparison using raw and inverse-ratio weighted MLH1."""
from pathlib import Path
import hashlib
import json

import anndata as ad
import h5py
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / '最终模型'
BASE = FINAL / 'data/scRNA/result/diagnostics'
OUT = BASE / 'shared18_inverse_weighted_mlh1_seed2026'
BRANCH = BASE / 'infercnvpy_stromal_only_unique_seed2026'
BENCH = BASE / 'shared18_same_sample_benchmark_seed2026'
TRAIN_MLH1 = BASE / 'infercnvpy_stromal_mlh1_impact_q95_training_unique_seed2026'
EXT_MLH1 = BASE / 'infercnvpy_stromal_mlh1_impact_q95_external_unique_seed2026'
MODEL = FINAL / 'model/reference/shared18_LDA_zscore2026/model.joblib'
MLH1 = 'mean_MLH1_epithelial'
SEED = 2026


def read_inputs(ds):
    source = ROOT / (ds + '.h5ad')
    if ds == 'GSE200997':
        source = FINAL / 'data/scRNA/external/GSE200997/GSE200997.h5ad'
    with h5py.File(source, 'r') as handle:
        obs = ad.io.read_elem(handle['obs'])
    assert obs.index.is_unique
    sample_col = 'sample' if ds == 'GSE178341' else 'Sample'
    if ds == 'GSE178341':
        epithelial = obs.clTopLevel.astype(str).eq('Epi')
    elif ds == 'GSE132465':
        epithelial = obs.Cell_type.astype(str).eq('Epithelial cells')
    else:
        migration = pd.read_csv(FINAL / 'data/scRNA/result/seed2026_migration_qc/GSE200997/process/GSE200997_tosica_migration_cells.csv').set_index('cell_id')
        assert migration.index.is_unique
        epithelial = migration.major_prediction.reindex(obs.index).eq('Epi')
    scores = pd.read_csv(BRANCH / ds / 'cell_malignancy_scores.csv.gz', index_col=0)
    assert scores.index.is_unique and scores.index.isin(obs.index).all()
    cells = scores[scores.Class.eq('Tumor')].copy()
    cells['epithelial'] = epithelial.reindex(cells.index)
    assert cells.Sample.eq(obs.loc[cells.index, sample_col].astype(str)).all()
    counts = cells.groupby('Sample').agg(n_total=('Class', 'size'), n_epithelial=('epithelial', 'sum'))
    counts.index.name = 'sampleID'
    directory = TRAIN_MLH1 if ds == 'GSE178341' else EXT_MLH1
    selected = pd.read_csv(directory / f'{ds}_selected_epithelial_cells.csv.gz', index_col=0)
    assert selected.index.is_unique and selected.index.isin(cells.index).all()
    assert cells.loc[selected.index, 'epithelial'].all()
    assert cells.loc[selected.index, 'cnv_reference'].isna().all()
    count_malignant = selected.groupby('Sample').size()
    audit = pd.read_csv(directory / f'{ds}_sample_MLH1_stromal_infercnvpy.csv').set_index('sampleID')
    counts = counts.reindex(audit.index)
    counts['n_malignant'] = count_malignant.reindex(counts.index, fill_value=0)
    np.testing.assert_array_equal(counts.n_malignant, audit.n_malignant_epithelial)
    np.testing.assert_allclose(selected.groupby('Sample').MLH1.mean().reindex(audit.index),
                               audit.mean_MLH1_epithelial_new, equal_nan=True)
    assert counts.n_total.ge(counts.n_epithelial).all()
    assert counts.n_epithelial.ge(counts.n_malignant).all()
    counts['epithelial_fraction'] = counts.n_epithelial / counts.n_total
    counts['malignant_fraction_within_epithelial'] = counts.n_malignant / counts.n_epithelial.replace(0, np.nan)
    counts['inverse_weight'] = counts.epithelial_fraction / counts.malignant_fraction_within_epithelial.replace(0, np.nan)
    counts['raw_MLH1'] = audit.mean_MLH1_epithelial_new
    counts['weighted_MLH1'] = counts.raw_MLH1 * counts.inverse_weight
    counts['MMR'] = audit.MMR
    counts['dataset'] = ds
    counts['count_scope'] = 'unique tumor cells retained in cohort CNV analysis'
    features_path = BENCH / f'{ds}_benchmark_features.csv'
    if ds == 'GSE200997':
        features_path = BASE / 'GSE200997_locked_shared18/benchmark_features.csv'
    features = pd.read_csv(features_path).set_index('sampleID').loc[audit.index].copy()
    assert features.index.is_unique and features.MMR.eq(counts.MMR).all()
    return features, counts


def evaluate(weighting='inverse'):
    OUT = BASE / f'shared18_{weighting}_weighted_mlh1_seed2026'
    formula = ('mean_MLH1 * N_malignant / N_epi' if weighting == 'malignant_fraction'
               else 'mean_MLH1 * N_epi**2 / (N_total*N_malignant)')
    OUT.mkdir(exist_ok=True)
    artifact = joblib.load(MODEL)
    features = list(artifact['features'])
    assert len(features) == 19 and features[-1] == MLH1
    datasets = ['GSE178341', 'GSE132465', 'GSE200997']
    inputs = {ds: read_inputs(ds) for ds in datasets}
    for _, count in inputs.values():
        count['applied_weight'] = (count.malignant_fraction_within_epithelial
                                   if weighting == 'malignant_fraction' else count.inverse_weight)
        count['weighted_MLH1'] = count.raw_MLH1 * count.applied_weight
    counts = pd.concat([a for _, a in inputs.values()])
    counts.to_csv(OUT / 'sample_counts_and_weights.csv')
    ledger = pd.read_csv(BENCH / 'sample_ledger.csv')
    ledger = ledger[ledger.dataset.eq('GSE178341')].set_index('sampleID')
    predictions = []
    for variant, column in [('raw_MLH1_refit', 'raw_MLH1'), (f'{weighting}_weighted_MLH1_refit', 'weighted_MLH1')]:
        tables = {}
        for ds, (table, count) in inputs.items():
            x = table.copy()
            x[MLH1] = count[column]
            assert np.isfinite(x[features].to_numpy()).all(), 'Missing feature: report coverage rather than fill zero'
            x[features].to_csv(OUT / f'{ds}_{variant}_features.csv')
            tables[ds] = x
        train = tables['GSE178341']
        y = train.MMR.eq('MSI').astype(int)
        meta = ledger.loc[train.index]
        assert meta.groupby('patientID').fold.nunique().eq(1).all()
        for fold in sorted(meta.fold.unique()):
            test = meta.fold.eq(fold)
            assert not set(meta.loc[test, 'patientID']) & set(meta.loc[~test, 'patientID'])
            model = clone(artifact['pipeline']).fit(train.loc[~test, features], y.loc[~test])
            prob = model.predict_proba(train.loc[test, features])[:, 1]
            for sid, score in zip(train.index[test], prob):
                predictions.append(dict(variant=variant, dataset='GSE178341', split='grouped_OOF',
                                        sampleID=sid, y=int(y.loc[sid]), score=score, fold=int(fold)))
        final = clone(artifact['pipeline']).fit(train[features], y)
        joblib.dump(dict(pipeline=final, features=features, seed=SEED, variant=variant,
                         target='MSI', formula=formula if column == 'weighted_MLH1' else 'raw_MLH1'),
                    OUT / f'{variant}.joblib')
        for ds, table in tables.items():
            prob = final.predict_proba(table[features])[:, 1]
            for sid, score in zip(table.index, prob):
                predictions.append(dict(variant=variant, dataset=ds,
                    split='training_apparent' if ds == 'GSE178341' else 'external',
                    sampleID=sid, y=int(table.loc[sid, 'MMR'] == 'MSI'), score=score, fold=0))
    predictions = pd.DataFrame(predictions)
    predictions['prediction_05'] = predictions.score.ge(.5).astype(int)
    predictions.to_csv(OUT / 'sample_predictions.csv', index=False)
    metrics = []
    for (variant, ds, split), table in predictions.groupby(['variant', 'dataset', 'split'], sort=False):
        tn, fp, fn, tp = confusion_matrix(table.y, table.prediction_05, labels=[0, 1]).ravel()
        metrics.append(dict(variant=variant, dataset=ds, split=split, n=len(table), MSI=int(table.y.sum()),
            AUROC=roc_auc_score(table.y, table.score), accuracy=accuracy_score(table.y, table.prediction_05),
            BA=balanced_accuracy_score(table.y, table.prediction_05), TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp)))
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(OUT / 'model_metrics.csv', index=False)
    five = predictions[predictions.dataset.eq('GSE132465') & predictions.sampleID.isin(
        ['SMC01-T', 'SMC02-T', 'SMC09-T', 'SMC17-T', 'SMC22-T'])].pivot(index='sampleID', columns='variant', values='score')
    five.to_csv(OUT / 'GSE132465_previous_five_false_positives.csv')
    (OUT / 'manifest.json').write_text(json.dumps(dict(seed=SEED, formula=formula,
        count_scope='unique tumor cells retained in each cohort CNV analysis; GSE178341 is sampled and enriched for reference cells by union',
        clipping='none', missing='zero malignant cells = missing; actual zero expression retained',
        evaluation='fixed historical18; training-fitted Z-score and lsqr auto-shrinkage LDA; existing patient-grouped folds; no tuning',
        limitations='cohort-wide CNV and historically selected features are not fully nested; external cohorts previously examined',
        reference=str(BRANCH), classifier_template=str(MODEL),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2))
    print(metrics.to_string(index=False))
    print(five.to_string())
    print(counts.groupby('dataset').applied_weight.agg(['min', 'median', 'max']).to_string())


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weighting', choices=['inverse', 'malignant_fraction'], default='inverse')
    evaluate(parser.parse_args().weighting)
