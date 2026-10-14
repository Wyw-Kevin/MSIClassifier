"""Independent saved-model, patient-split, baseline and pairwise-AUROC verification."""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
import joblib
from scipy.special import expit

HERE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = json.loads((HERE/'source_manifest.json').read_text())
protocol = json.loads((HERE/'protocol.json').read_text())
assert protocol['seed'] == 2026 and not protocol['external_labels_used_for_training_or_selection']
checks = 0
def check(value, label):
    global checks
    assert value, label
    checks += 1

pred = pd.read_csv(HERE/'all_predictions.csv', float_precision='round_trip')
metrics = pd.read_csv(HERE/'tables/all_metrics.csv', float_precision='round_trip')
index = pd.read_csv(HERE/'model_index.csv', dtype={'fold': str})
data = {ds: pd.read_csv(HERE/f'inputs/{ds}_q95_features_labels_folds.csv', float_precision='round_trip')
        for ds in ['GSE178341','GSE132465','GSE200997']}
dev = data['GSE178341']
check(len(index) == 42 and len(pred) == 1660, '42 actual new fits, 1660 saved predictions')
check(not pred.duplicated(['configuration','dataset','split','sampleID']).any(), 'Unique score records')
check(dev.patientID.nunique() == 62, 'Actual patient identities')
check(dev.groupby('patientID').fold.nunique().eq(1).all(), 'Patients never split across folds')
q = pd.read_csv(HERE/'inputs/CNV_quantile_sample_features.csv', float_precision='round_trip')
def frame_for(ds, quantile):
    frame = data[ds].copy()
    measurements = q.loc[q.dataset.eq(ds) & q['quantile'].eq(quantile)].set_index('sampleID').loc[frame.sampleID]
    check(measurements.n_candidate.gt(0).all() and measurements.n_epithelial.gt(0).all(), 'No missing malignant estimates')
    check(np.allclose(measurements.W, measurements.mu*measurements.n_candidate/measurements.n_epithelial, atol=2e-12, rtol=0), 'Independent weighting identity')
    if quantile != .95:
        frame.weighted_MLH1 = measurements.W.to_numpy()
    return frame

models = {}
for row in index.itertuples():
    p = HERE/row.path
    check(sha(p) == row.sha256, 'Actual model hash')
    artifact = joblib.load(p)
    check(artifact['seed'] == 2026 and artifact['training_dataset'] == 'GSE178341', 'Development-only fit provenance')
    fold = row.fold
    frame = frame_for('GSE178341', artifact['config']['CNV_quantile'])
    training = frame if fold == 'full' else frame.loc[frame.fold.ne(int(fold))]
    check(artifact['training_samples'] == training.sampleID.tolist(), 'Exact training samples')
    check(artifact['training_patients'] == training.patientID.tolist(), 'Exact training patients')
    if fold != 'full':
        test = frame.loc[frame.fold.eq(int(fold))]
        check(not set(training.patientID)&set(test.patientID), 'Patient holdout isolation')
    X = training[artifact['features']].to_numpy()
    scaler = artifact['pipeline'].named_steps['scaler']
    check(np.allclose(scaler.mean_, X.mean(axis=0), rtol=0, atol=1e-12), 'Scaler is fitted to development training rows')
    check(np.allclose(scaler.var_, X.var(axis=0), rtol=0, atol=1e-12), 'Scaler variance is development-only')
    lda = artifact['pipeline'].named_steps['model']
    check(lda.solver == 'lsqr' and lda.shrinkage == 'auto', 'Fixed LDA parameters')
    models[(artifact['config']['configuration'], fold)] = artifact

max_replay = 0.
for (configuration, ds, split, fold), rows in pred.groupby(['configuration','dataset','split','fold']):
    selected_model = configuration if rows.iloc[0]['mode'] == 'refit' else 'CNV_q95_refit'
    artifact = models[(selected_model, str(fold))]
    frame = frame_for(ds, float(rows.iloc[0].CNV_quantile)).set_index('sampleID').loc[rows.sampleID]
    check(np.array_equal(frame.y, rows.y), 'Saved score labels aligned')
    X = frame[artifact['features']].to_numpy()
    replay = artifact['pipeline'].predict_proba(X)[:,1]
    error = float(np.max(np.abs(replay-rows.score)))
    max_replay = max(max_replay,error)
    check(error < 1e-12, 'Every saved score replays its actual model')
    scaler = artifact['pipeline'].named_steps['scaler']
    lda = artifact['pipeline'].named_steps['model']
    manual = expit(((X-scaler.mean_)/scaler.scale_)@lda.coef_.ravel()+lda.intercept_[0])
    check(np.allclose(manual,rows.score,rtol=0,atol=1e-12), 'Independent scaler/linear decision/probability calculation')

for row in metrics.itertuples():
    part = pred.loc[pred.configuration.eq(row.configuration)&pred.dataset.eq(row.dataset)&pred.split.eq(row.split)]
    pos, neg = part.loc[part.y.eq(1),'score'].to_numpy(), part.loc[part.y.eq(0),'score'].to_numpy()
    auc = float(((pos[:,None]>neg[None,:])+.5*(pos[:,None]==neg[None,:])).mean())
    check(abs(auc-row.AUROC) < 1e-12, 'Independent pairwise AUROC')
    ba = .5*((pos>=.5).mean()+(neg<.5).mean())
    check(abs(ba-row.BA) < 1e-12, 'Independent balanced accuracy')
    check(abs(np.square(part.score-part.y).mean()-row.Brier) < 1e-12, 'Independent Brier score')
    check(len(part) == row.n_samples, 'Reported denominators')

base_errors = {}
for ds in data:
    split = 'grouped_OOF' if ds == 'GSE178341' else 'external'
    old = data[ds].set_index('sampleID')
    for configuration in ['CNV_q95_refit','CNV_q95_frozen_q95']:
        new = pred.loc[pred.configuration.eq(configuration)&pred.dataset.eq(ds)&pred.split.eq(split)].set_index('sampleID').loc[old.index]
        error = float(np.max(np.abs(new.score-old.archived_score)))
        base_errors[configuration+'/'+ds] = error
        check(error < 1e-10, 'Current q95/main-panel score baseline reproduced')
current_path = Path(manifest['current_model']['path'])
if not current_path.exists():
    current_path = HERE/'inputs/source_archive/current_model.joblib'
current = joblib.load(current_path)
for ds in data:
    frame = data[ds]
    # The original checkpoint uses a legacy feature name; its numeric last column is weighted MLH1.
    X = frame[current['features'][:-1]+['weighted_MLH1']].to_numpy()
    X = pd.DataFrame(X, columns=current['features'])
    replay = current['pipeline'].predict_proba(X)[:,1]
    new = pred.loc[pred.configuration.eq('CNV_q95_refit')&pred.dataset.eq(ds)
                   &pred.split.eq('training_apparent' if ds=='GSE178341' else 'external')].set_index('sampleID').loc[frame.sampleID]
    check(np.allclose(replay,new.score,rtol=0,atol=1e-10), 'Archived full classifier baseline reproduced')

membership = pd.read_csv(HERE/'tables/subtype_threshold_membership.csv')
ledger = pd.read_csv(HERE/'inputs/candidate_ledger.csv')
reference = pd.read_csv(HERE/'inputs/reference71_features.csv').set_index('sampleID')
for r in ledger.itertuples():
    s = reference.loc[dev.sampleID,r.feature].to_numpy()
    p, n = s[dev.y.to_numpy()==1], s[dev.y.to_numpy()==0]
    raw = float(((p[:,None]>n[None,:])+.5*(p[:,None]==n[None,:])).mean())
    check(abs(raw-r.auc_msi_high)<1e-12 and abs(max(raw,1-raw)-r.auc_best_direction)<1e-12, 'Independent reference ranking AUROC')
for cutoff, expected in zip([.60,.65,.70,.75],[39,24,17,9]):
    rows = membership.loc[membership.screening_cutoff.eq(cutoff)]
    check(rows.passes_screen.sum()==expected and rows.in_model.sum()==expected, 'All passing candidates enter; no cap')
    check(np.array_equal(rows.passes_screen,rows.reference_oriented_AUROC.gt(cutoff)), 'Strict cutoff')
    check(np.array_equal(rows.passes_screen, rows.in_model), 'Membership equals threshold-only selection')
display = pd.read_csv(HERE/'tables/S10_threshold_performance.csv')
check(len(display)==15 and display.family.eq('fixed_panel_reference').sum()==3, 'Historical18 separately labeled in display table')
check(display.loc[display.family.eq('subtype_threshold')].n_subtypes.equals(display.loc[display.family.eq('subtype_threshold')].n_candidate_above_cutoff), 'Uncapped display denominators')
for item in index.itertuples():
    artifact = models[(item.configuration, item.fold)]
    if artifact['config']['family'] == 'subtype_threshold':
        expected = ledger.loc[ledger.auc_best_direction.gt(artifact['config']['screening_cutoff']), 'feature'].tolist()
        check(set(artifact['features'][:-1]) == set(expected), 'Actual model includes every passing candidate')
ranges = pd.read_csv(HERE/'tables/AUROC_sensitivity_ranges.csv')
reported = metrics.loc[metrics.split.ne('training_apparent')]
for r in ranges.itertuples():
    part = reported.loc[reported.family.eq(r.family)&reported['mode'].eq(r.mode)&reported.dataset.eq(r.dataset)]
    check(abs(part.AUROC.min()-r.AUROC_min)<1e-12 and abs(part.AUROC.max()-r.AUROC_max)<1e-12, 'Every reported sensitivity range')
for key, item in manifest.items():
    path = Path(item['path'])
    if not path.exists():
        path = HERE/'inputs/source_archive'/(key+path.suffix)
    check(sha(path) == item['sha256'], 'Original source unchanged: '+key)
result = {'status': 'passed', 'checks': checks, 'fitted_models_replayed': len(index),
          'predictions_verified': len(pred), 'pairwise_metrics_verified': len(metrics),
          'max_saved_prediction_replay_error': max_replay,
          'max_primary_baseline_score_error': max(base_errors.values()),
          'baseline_errors': base_errors, 'all_original_source_hashes_unchanged': True,
          'all_patient_holdout_and_training_only_scaler_checks': True,
          'current_primary_model_replaced': False,
          'interpretation': 'Conditional sensitivity with fixed historical ranking and cohort-level CNV; not untouched external validation'}
(HERE/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False,indent=2))
