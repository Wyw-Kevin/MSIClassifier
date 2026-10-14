"""Fresh Z-score audits; fixed18 LDA and reselection are distinct estimands."""
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import run_scrna_metadata_adjusted_pipeline as core
import scrna_clinical_ablation as ab
from run_scrna_age_sex_sensitivity import add_literature_age_group

OUT = core.FINAL / 'data/scRNA/result/diagnostics/zscore_clinical_ablation_seed2026'
TRACKS = {'expression_only': ab.ClinicalTrack((), ()),
          'age_continuous_sex': ab.ClinicalTrack(('age',), ('female',)),
          'age_group50_sex': ab.ClinicalTrack((), ('early_onset_lt50', 'female'))}


def load_data():
    train = pd.read_csv(core.TRAIN_FEATURES_PATH)
    external = pd.read_csv(core.EXTERNAL_FEATURES_PATH)
    candidates = pd.read_csv(core.CLASS_CONTRACT_PATH).class_name.tolist()
    mt, me = core._validate_production_contract(train, external,
        add_literature_age_group(core.parse_gse178341_metadata(core.GSE178341_METADATA_PATH)),
        add_literature_age_group(core.parse_gse132465_metadata(core.GSE132465_METADATA_PATH)), candidates)
    mt = core._align_metadata(train, mt, 'train')
    me = core._align_metadata(external, me, 'external')
    folds = core.make_patient_grouped_folds(mt.y, mt.patientID, n_splits=5, random_state=2026)
    for fit, test in folds:
        assert not set(mt.iloc[fit].patientID) & set(mt.iloc[test].patientID)
    return train, external, mt, me, candidates, folds


def frame(data, meta, score, split):
    return pd.DataFrame({'sampleID': data.sampleID.to_numpy(), 'patientID': meta.patientID.to_numpy(),
                         'y': core._binary_labels(data, split), 'score': score, 'split': split})


def fixed_job(track_name):
    train, external, mt, me, _, folds = load_data()
    track = TRACKS[track_name]
    model = core.make_models(2026)['lda_shrinkage']
    parts = []
    for configuration, features in {
        'full': list(core.CURRENT_SHARED18) + [core.MLH1_FEATURE],
        'ecology_only': list(core.CURRENT_SHARED18), 'MLH1_only': [core.MLH1_FEATURE]}.items():
        oof = np.full(len(train), np.nan)
        for fit, test in folds:
            oof[test] = ab._fit_selected_and_score(train.iloc[fit], train.iloc[test], mt.iloc[fit], mt.iloc[test],
                features, model, track, 2026, normalization='zscore')
        parts.append(frame(train, mt, oof, 'train_grouped_oof').assign(configuration=configuration))
        for split, data, meta in [('train_full_fit', train, mt), ('external', external, me)]:
            score = ab._fit_selected_and_score(train, data, mt, meta, features, model, track, 2026, normalization='zscore')
            parts.append(frame(data, meta, score, split).assign(configuration=configuration))
    if track.adjusted:
        _, clinical = core.evaluate_clinical_baseline(train, external, mt, me, folds, list(track.covariates), 2026)
        parts.append(clinical.assign(configuration='clinical_only'))
    result = pd.concat(parts, ignore_index=True).assign(protocol='fixed18_LDA', clinical_track=track_name)
    result.to_csv(OUT / f'fixed_{track_name}_scores.csv', index=False)
    return f'fixed_{track_name}'


def nested_job(track_name, configuration):
    train, external, mt, me, candidates, folds = load_data()
    track = TRACKS[track_name]
    models = core.make_models(2026)
    directory = OUT / f'nested_{track_name}_{configuration}'
    directory.mkdir(exist_ok=True)
    selected_parts, candidate_parts, prediction_parts = [], [], []
    for fold, (fit, test) in enumerate(folds, 1):
        tr, tm = train.iloc[fit].reset_index(drop=True), mt.iloc[fit].reset_index(drop=True)
        selection = ab.select_inner_candidate(tr, tm, candidates, models, track, configuration,
                                              .65, 10, 24, 5, 2026, normalization='zscore')
        if not selection.failures.empty:
            selection.failures.to_csv(directory / f'fold{fold}_failures.csv', index=False)
            raise RuntimeError(f'Candidate failure in {directory}, fold {fold}')
        features = list(selection.subtype_features) + ([core.MLH1_FEATURE] if configuration == 'ecology_plus_MLH1' else [])
        score = ab._fit_selected_and_score(tr, train.iloc[test], tm, mt.iloc[test], features,
            models[selection.model_name], track, 2026, normalization='zscore')
        prediction_parts.append(frame(train.iloc[test], mt.iloc[test], score, 'train_nested_oof').assign(fold=fold))
        selected_parts.append({'fold': fold, 'model': selection.model_name, 'features': ';'.join(selection.subtype_features),
                               'n_features': len(selection.subtype_features), 'inner_auc': selection.inner_auc})
        candidate_parts.append(selection.inner_candidates.assign(fold=fold))
        pd.DataFrame(selected_parts).to_csv(directory / 'fold_selections.csv', index=False)
        print(f'{track_name}/{configuration}: outer fold {fold}/5', flush=True)
    lock = ab.select_inner_candidate(train, mt, candidates, models, track, configuration,
                                     .65, 10, 24, 5, 2026, normalization='zscore')
    assert lock.failures.empty
    candidate_parts.append(lock.inner_candidates.assign(fold=0))
    features = list(lock.subtype_features) + ([core.MLH1_FEATURE] if configuration == 'ecology_plus_MLH1' else [])
    for split, data, meta in [('train_full_fit', train, mt), ('external', external, me)]:
        score = ab._fit_selected_and_score(train, data, mt, meta, features,
            models[lock.model_name], track, 2026, normalization='zscore')
        prediction_parts.append(frame(data, meta, score, split).assign(fold=0))
    result = pd.concat(prediction_parts, ignore_index=True).assign(protocol='reselected',
        clinical_track=track_name, configuration='full' if configuration == 'ecology_plus_MLH1' else 'ecology_only')
    result.to_csv(directory / 'scores.csv', index=False)
    pd.concat(candidate_parts, ignore_index=True).to_csv(directory / 'inner_candidates.csv', index=False)
    (directory / 'lock.json').write_text(json.dumps({'model': lock.model_name, 'features': lock.subtype_features,
        'seed': 2026, 'normalization': 'zscore', 'inner_auc': lock.inner_auc}, indent=2))
    return directory.name


def summarize():
    tables = [pd.read_csv(p) for p in OUT.glob('fixed_*_scores.csv')]
    tables += [pd.read_csv(p) for p in OUT.glob('nested_*/scores.csv')]
    all_scores = pd.concat(tables, ignore_index=True)
    rows, pairs, reps = [], [], []
    keys = ['protocol', 'clinical_track', 'configuration', 'split']
    for key, part in all_scores.groupby(keys):
        assert part.sampleID.is_unique and np.isfinite(part.score).all()
        assert len(part) == (23 if key[-1] == 'external' else 64)
        auc, low, high, success = core.stratified_bootstrap_auc(part.y, part.score, part.patientID, 2000, 2026)
        assert np.isclose(auc, roc_auc_score(part.y, part.score))
        rows.append({**dict(zip(keys, key)), 'n': len(part), 'auc': auc, 'ci_lower': low, 'ci_upper': high,
                     'bootstrap_success': success, 'random_state': 2026})
    for key, group in all_scores.groupby(['protocol', 'clinical_track', 'split']):
        full = group[group.configuration.eq('full')].sort_values('sampleID')
        if full.empty or key[-1] == 'train_full_fit':
            continue
        for comparator in ['ecology_only', 'MLH1_only']:
            other = group[group.configuration.eq(comparator)].sort_values('sampleID')
            if other.empty:
                continue
            assert full.sampleID.tolist() == other.sampleID.tolist()
            assert full.y.tolist() == other.y.tolist()
            summary, replicate = ab.paired_patient_bootstrap_auc_delta_with_replicates(
                full.y, full.score, other.score, full.patientID, 2000, 2026)
            tags = dict(zip(['protocol', 'clinical_track', 'split'], key))
            tags.update(comparator=comparator, random_state=2026)
            pairs.append({**tags, **summary})
            reps.append(replicate.assign(**tags))
    all_scores.to_csv(OUT / 'all_predictions.csv', index=False)
    pd.DataFrame(rows).to_csv(OUT / 'AUROC_summary.csv', index=False)
    pd.DataFrame(pairs).to_csv(OUT / 'paired_ablation.csv', index=False)
    pd.concat(reps, ignore_index=True).to_csv(OUT / 'paired_bootstrap_replicates.csv.gz', index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Workers use separate files and no historical prediction cache.
    with ProcessPoolExecutor(max_workers=6) as pool:
        jobs = [pool.submit(fixed_job, track) for track in TRACKS]
        jobs += [pool.submit(nested_job, track, config) for track in TRACKS
                 for config in ['ecology_plus_MLH1', 'ecology_only']]
        for future in as_completed(jobs):
            print('Completed', future.result(), flush=True)
    summarize()


if __name__ == '__main__':
    main()
