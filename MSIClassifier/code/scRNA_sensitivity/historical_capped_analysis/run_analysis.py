"""Conditional scRNA subtype-threshold and stromal-only CNV quantile sensitivity."""
from pathlib import Path
from copy import deepcopy
import hashlib, json, sys, platform, shutil
import numpy as np
import pandas as pd
import joblib, sklearn
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, brier_score_loss

HERE = Path(__file__).resolve().parent
FINAL = HERE.parents[2]
ARTICLE = HERE.parent
DIAG = FINAL/'data/scRNA/result/diagnostics'
PREVIOUS = ARTICLE/'第三版_十亚型与留一队列MMR4修订_研究设计图恢复版/补充图S2十亚型数量入选率'
SUP = PREVIOUS/'supplement'
DS = ['GSE178341', 'GSE132465', 'GSE200997']
THRESHOLDS = [.60, .65, .70, .75]
QUANTILES = [.90, .95, .99]
W = 'weighted_MLH1'
for directory in ['inputs', 'models', 'tables']:
    (HERE/directory).mkdir(exist_ok=True)
sources = {
    'current_model': FINAL/'model/reference/shared18_weighted_mlh1_2026/model.joblib',
    'current_manifest': FINAL/'model/reference/shared18_weighted_mlh1_2026/manifest.json',
    'candidate_ledger': SUP/'S10.csv',
    'cnv_quantile_summary': SUP/'S2.csv',
    'cnv_sample_features': SUP/'S3.csv',
    'patient_ledger': DIAG/'shared18_same_sample_benchmark_seed2026/sample_ledger.csv',
    'fold_manifest': DIAG/'shared18_same_sample_benchmark_seed2026/manifest.json',
    'current_predictions': ARTICLE/'weighted_scRNA_update/all_model_predictions.csv',
    'reference71': DIAG/'unmerged_raw_subtypes_min50_20260908/GSE178341_unmerged71_features.csv',
}
for ds in DS:
    sources[ds+'_features'] = DIAG/f'shared18_malignant_fraction_weighted_mlh1_seed2026/{ds}_malignant_fraction_weighted_MLH1_refit_features.csv'
archive_dir = HERE/'inputs/source_archive'
archive_dir.mkdir(exist_ok=True)
for key, path in list(sources.items()):
    archived = archive_dir/(key+path.suffix)
    if path.exists():
        shutil.copy2(path, archived)
    else:
        assert archived.exists(), key
        sources[key] = archived
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
source_manifest = {key: {'path': str(path), 'sha256': sha(path)} for key, path in sources.items()}
model = joblib.load(sources['current_model'])
assert model['seed'] == 2026
assert json.loads(sources['fold_manifest'].read_text())['seed'] == 2026
panel = model['features'][:-1]
legacy_w = model['features'][-1]
assert legacy_w == 'mean_MLH1_epithelial' and len(panel) == 18
ledger = pd.read_csv(sources['candidate_ledger'], float_precision='round_trip')
qfeatures = pd.read_csv(sources['cnv_sample_features'], float_precision='round_trip')
patients = pd.read_csv(sources['patient_ledger'])
reference = pd.read_csv(sources['reference71'], float_precision='round_trip').set_index('sampleID')
previous_predictions = pd.read_csv(sources['current_predictions'], float_precision='round_trip')
ranked = ledger.sort_values('auc_best_direction', ascending=False, kind='stable')
assert ranked.head(18).feature.tolist() == panel
assert len(ledger) == 71 and ledger.feature.is_unique
assert int(ledger.selected_24.sum()) == 24
assert set(panel) == set(ledger.loc[ledger.retained_fixed18, 'feature'])

# Lock the descriptive comparison before fitting/scoring any new configuration.
protocol = {
    'seed': 2026,
    'status': 'locked before new classifier fitting and evaluation',
    'main_rule_unchanged': {'subtype_cutoff': .65, 'maximum_subtypes': 18, 'CNV_quantile': .95},
    'subtype_thresholds': THRESHOLDS,
    'subtype_rule': 'Frozen full-development historical best-direction AUROC strictly above cutoff; descending historical ranking; cap at 18',
    'CNV_quantiles': QUANTILES,
    'CNV_scope': 'Saved separate-cohort stromal-only CNV scores and same retained epithelial denominators; no new CNV or cell annotation',
    'CNV_modes': ['Re-fit scaler and shrinkage LDA for each quantile', 'Freeze q95 fold/full models and change only weighted MLH1 inputs'],
    'sampling_and_folds': 'Reuse archived patient-grouped five folds: 64 samples/62 patients; no splitting RNG or new bootstrap',
    'fit': 'StandardScaler and lsqr auto-shrinkage LDA; fit only GSE178341 training fold or full development set',
    'external_cohorts': ['GSE132465', 'GSE200997'],
    'external_labels_used_for_training_or_selection': False,
    'upstream_status': 'Historical candidate ranking and cohort-wide CNV remain fixed; conditional sensitivity, not nested end-to-end evaluation',
    'missing': 'Zero candidate malignant cells means missing; preserve observed zero MLH1; evaluate scoreable sample set and disclose differences',
    'input_alias': legacy_w+' in archived features contains malignant-fraction-weighted MLH1, renamed weighted_MLH1 here',
    'primary_checkpoint_replacement': False,
    'threshold_selection_from_new_results': False,
    'software': {'python': sys.version.split()[0], 'numpy': np.__version__, 'pandas': pd.__version__,
                 'scikit_learn': sklearn.__version__, 'platform': platform.platform()},
}
(HERE/'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2))
(HERE/'source_manifest.json').write_text(json.dumps(source_manifest, ensure_ascii=False, indent=2))
data = {}
for ds in DS:
    features = pd.read_csv(sources[ds+'_features'], float_precision='round_trip').rename(columns={legacy_w: W})
    pred = previous_predictions.loc[previous_predictions.dataset.eq(ds)
          & previous_predictions.model.eq('18 subtypes + weighted MLH1')
          & previous_predictions.split.eq('grouped_OOF' if ds == DS[0] else 'external')]
    assert len(pred) == len(features) and pred.sampleID.is_unique
    meta = pred[['sampleID', 'y', 'fold', 'score']].rename(columns={'score': 'archived_score'})
    frame = features.merge(meta, on='sampleID', validate='one_to_one', sort=False)
    if ds == DS[0]:
        p = patients.loc[patients.dataset.eq(ds), ['sampleID', 'patientID', 'fold', 'y']]
        check = frame.merge(p, on='sampleID', validate='one_to_one', suffixes=('', '_patient'))
        assert check.y.eq(check.y_patient).all() and check.fold.eq(check.fold_patient).all()
        frame['patientID'] = check.patientID.to_numpy()
        assert frame.groupby('patientID').fold.nunique().eq(1).all()
        assert frame.patientID.nunique() == 62 and len(frame) == 64
        assert set(frame.fold) == set(range(1, 6))
        assert set(frame.sampleID) == set(reference.index)
        assert np.allclose(frame[panel], reference.loc[frame.sampleID, panel], atol=2e-12, rtol=0)
    else:
        frame['patientID'] = frame.sampleID
    frame.insert(0, 'dataset', ds)
    assert np.isfinite(frame[panel+[W]].to_numpy()).all()
    q95 = qfeatures.loc[qfeatures.dataset.eq(ds) & qfeatures['quantile'].eq(.95)].set_index('sampleID')
    assert np.allclose(frame[W], q95.loc[frame.sampleID, 'W'], atol=2e-12, rtol=0)
    assert len(frame) == {'GSE178341':64, 'GSE132465':23, 'GSE200997':15}[ds]
    assert int(frame.y.sum()) == {'GSE178341':35, 'GSE132465':4, 'GSE200997':2}[ds]
    data[ds] = frame
    frame.to_csv(HERE/f'inputs/{ds}_q95_features_labels_folds.csv', index=False)
ledger.to_csv(HERE/'inputs/candidate_ledger.csv', index=False)
reference[ledger.feature].to_csv(HERE/'inputs/reference71_features.csv')
qfeatures.to_csv(HERE/'inputs/CNV_quantile_sample_features.csv', index=False)
pd.read_csv(sources['cnv_quantile_summary']).to_csv(HERE/'inputs/CNV_quantile_summary.csv', index=False)

# Audit the selection ranking against the original development proportions.
for row in ledger.itertuples():
    raw_auc = roc_auc_score(data[DS[0]].y, reference.loc[data[DS[0]].sampleID, row.feature])
    assert abs(raw_auc-row.auc_msi_high) < 1e-12
    assert abs(max(raw_auc, 1-raw_auc)-row.auc_best_direction) < 1e-12
membership = []
sets = {}
for cutoff in THRESHOLDS:
    candidates = ranked.loc[ranked.auc_best_direction.gt(cutoff)]
    features = candidates.head(18).feature.tolist()
    assert set(features).issubset(panel)
    sets[cutoff] = features
    for row in ledger.itertuples():
        membership.append({'screening_cutoff': cutoff, 'feature': row.feature,
            'reference_oriented_AUROC': row.auc_best_direction,
            'passes_screen': row.auc_best_direction > cutoff, 'in_model': row.feature in features,
            'rank': int(ranked.feature.tolist().index(row.feature)+1)})
pd.DataFrame(membership).to_csv(HERE/'tables/subtype_threshold_membership.csv', index=False)

predictions, fit_records = [], []
def predict_rows(pipe, frame, cols, config, split, fold):
    score = pipe.predict_proba(frame[cols].to_numpy())[:, 1]
    for sample, y, s in zip(frame.sampleID, frame.y, score):
        predictions.append({**config, 'dataset': frame.dataset.iloc[0], 'sampleID': sample,
                            'y': int(y), 'split': split, 'fold': fold, 'score': float(s),
                            'prediction_05': int(s >= .5), 'n_subtypes': len(cols)-1})

def fit_model(frame, cols, config, fold):
    pipe = clone(model['pipeline'])
    pipe.fit(frame[cols].to_numpy(), frame.y.to_numpy())
    name = f"{config['configuration']}__fold{fold}.joblib"
    artifact = {'pipeline': pipe, 'features': cols, 'config': config, 'seed': 2026,
                'fold': fold, 'training_dataset': DS[0],
                'training_samples': frame.sampleID.tolist(), 'training_patients': frame.patientID.tolist()}
    joblib.dump(artifact, HERE/'models'/name)
    fit_records.append({'configuration': config['configuration'], 'fold': str(fold),
                        'n_training_samples': len(frame), 'n_training_patients': frame.patientID.nunique(),
                        'features': ';'.join(cols), 'path': 'models/'+name,
                        'sha256': sha(HERE/'models'/name), 'training_dataset': DS[0], 'seed': 2026})
    return pipe

def refit_configuration(config, features, frames):
    cols = features+[W]
    dev = frames[DS[0]]
    pipes = {}
    for fold in range(1, 6):
        train, test = dev.loc[dev.fold.ne(fold)], dev.loc[dev.fold.eq(fold)]
        assert not set(train.patientID) & set(test.patientID)
        assert train.y.nunique() == test.y.nunique() == 2
        pipe = fit_model(train, cols, config, fold)
        predict_rows(pipe, test, cols, config, 'grouped_OOF', fold)
        pipes[fold] = pipe
    pipe = fit_model(dev, cols, config, 'full')
    pipes['full'] = pipe
    predict_rows(pipe, dev, cols, config, 'training_apparent', 'full')
    for ds in DS[1:]:
        predict_rows(pipe, frames[ds], cols, config, 'external', 'full')
    return pipes

def cnv_frames(quantile):
    frames = {}
    for ds in DS:
        frame = data[ds].copy()
        q = qfeatures.loc[qfeatures.dataset.eq(ds) & qfeatures['quantile'].eq(quantile)].set_index('sampleID')
        assert set(q.index) == set(frame.sampleID)
        q = q.loc[frame.sampleID]
        assert q.scoreable.all() and q.n_candidate.gt(0).all() and q.n_epithelial.gt(0).all()
        assert np.allclose(q.W, q.mu*q.n_candidate/q.n_epithelial, atol=2e-12, rtol=0)
        if quantile != .95:
            frame[W] = q.W.to_numpy()
        frame.to_csv(HERE/f'inputs/{ds}_q{int(quantile*100)}_features.csv', index=False)
        frames[ds] = frame
    return frames

for cutoff in THRESHOLDS:
    config = {'configuration': f'subtype_cutoff_{int(cutoff*100):02}', 'family': 'subtype_threshold',
              'screening_cutoff': cutoff, 'CNV_quantile': .95, 'mode': 'refit'}
    refit_configuration(config, sets[cutoff], data)
cnv_data = {q: cnv_frames(q) for q in QUANTILES}
q95_models = None
# Train the fixed main rule first so its folds can also be reused for input perturbations.
for quantile in [.95, .90, .99]:
    config = {'configuration': f'CNV_q{int(quantile*100)}_refit', 'family': 'CNV_quantile',
              'screening_cutoff': .65, 'CNV_quantile': quantile, 'mode': 'refit'}
    pipes = refit_configuration(config, panel, cnv_data[quantile])
    if quantile == .95:
        q95_models = pipes
for quantile in QUANTILES:
    config = {'configuration': f'CNV_q{int(quantile*100)}_frozen_q95', 'family': 'CNV_quantile',
              'screening_cutoff': .65, 'CNV_quantile': quantile, 'mode': 'frozen_q95'}
    frames = cnv_data[quantile]
    for fold in range(1, 6):
        predict_rows(q95_models[fold], frames[DS[0]].loc[frames[DS[0]].fold.eq(fold)],
                     panel+[W], config, 'grouped_OOF', fold)
    for ds in DS:
        predict_rows(q95_models['full'], frames[ds], panel+[W], config,
                     'training_apparent' if ds == DS[0] else 'external', 'full')
pred = pd.DataFrame(predictions)
pred.to_csv(HERE/'all_predictions.csv', index=False)
pd.DataFrame(fit_records).to_csv(HERE/'model_index.csv', index=False)
rows = []
group = ['configuration','family','screening_cutoff','CNV_quantile','mode','n_subtypes','dataset','split']
for keys, part in pred.groupby(group):
    row = dict(zip(group, keys))
    y, s = part.y.to_numpy(), part.score.to_numpy()
    row.update(n_samples=len(part), n_positive=int(y.sum()), n_negative=int((1-y).sum()),
               AUROC=roc_auc_score(y, s), BA=balanced_accuracy_score(y, s>=.5), Brier=brier_score_loss(y, s))
    rows.append(row)
metrics = pd.DataFrame(rows)
metrics.to_csv(HERE/'tables/all_metrics.csv', index=False)
reported = metrics.loc[metrics.split.ne('training_apparent')].copy()
reported.to_csv(HERE/'tables/reported_metrics.csv', index=False)
screen = reported.loc[reported.family.eq('subtype_threshold')].copy()
screen['n_candidate_above_cutoff'] = screen.screening_cutoff.map({x: int(ranked.auc_best_direction.gt(x).sum()) for x in THRESHOLDS})
screen.to_csv(HERE/'tables/S10_threshold_performance.csv', index=False)
reported.loc[reported.family.eq('CNV_quantile')].to_csv(HERE/'tables/S3_CNV_quantile_performance.csv', index=False)
ranges = []
for (family, mode, ds, split), part in reported.groupby(['family','mode','dataset','split']):
    baseline = part.loc[part.screening_cutoff.eq(.65)] if family == 'subtype_threshold' else part.loc[part.CNV_quantile.eq(.95)]
    assert len(baseline) == 1
    ref_auc = float(baseline.iloc[0].AUROC)
    ranges.append({'family': family, 'mode': mode, 'dataset': ds, 'split': split,
                   'n_samples': int(part.iloc[0].n_samples), 'baseline_AUROC': ref_auc,
                   'AUROC_min': float(part.AUROC.min()), 'AUROC_max': float(part.AUROC.max()),
                   'AUROC_span': float(part.AUROC.max()-part.AUROC.min()),
                   'delta_vs_baseline_min': float(part.AUROC.min()-ref_auc),
                   'delta_vs_baseline_max': float(part.AUROC.max()-ref_auc)})
pd.DataFrame(ranges).to_csv(HERE/'tables/AUROC_sensitivity_ranges.csv', index=False)
print(reported[['configuration','dataset','n_subtypes','n_samples','AUROC']].to_string(index=False))
assert all(sha(path) == source_manifest[key]['sha256'] for key,path in sources.items())
(HERE/'calculation_complete.json').write_text(json.dumps({'status': 'calculated; requires independent verification',
    'fitted_models': len(fit_records), 'prediction_rows': len(pred),
    'primary_models_and_source_files_unchanged': True}, indent=2))
