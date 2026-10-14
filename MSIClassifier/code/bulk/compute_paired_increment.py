"""Conditional paired fusion-minus-LOCO-MMR4 intervals; no model refit."""
from pathlib import Path
import argparse, hashlib, json, shutil
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'bulk_7_vs12_mlp_mmr4_loco_seed2026_20261013'
LOCO = HERE.parent / 'bulk_mmr4_loco_fixed05_seed2026_20261012'
OUT = HERE / 'paired_increment'
DEV = ['GSE39582', 'TCGA-COADREAD']
GENES = ['MLH1', 'MSH2', 'MSH6', 'PMS2']
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p, v): p.write_text(json.dumps(v, ensure_ascii=False, indent=2) + '\n')

def weighted_pair_auc(y, score, indices):
    """Independent AUC evaluator using all original positive-negative pairs."""
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    counts = np.zeros((len(indices), len(y)), dtype=np.int64)
    np.add.at(counts, (np.repeat(np.arange(len(indices)), indices.shape[1]), indices.ravel()), 1)
    kernel = (score[pos, None] > score[None, neg]).astype(float)
    kernel += 0.5 * (score[pos, None] == score[None, neg])
    return np.einsum('bi,ij,bj->b', counts[:, pos], kernel, counts[:, neg], optimize=True) / (len(pos) * len(neg))

def main():
    global SOURCE, LOCO, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay', action='store_true', help='Use preserved input copies without historical project folders')
    parser.add_argument('--output', type=Path, help='Write deterministic recomputation to a new directory')
    args = parser.parse_args()
    if args.replay:
        SOURCE = HERE / 'paired_increment/source_archive/model'
        LOCO = HERE / 'paired_increment/source_archive/mmr4'
    if args.output: OUT = args.output
    OUT.mkdir(exist_ok=True)
    source_files = [SOURCE / n for n in ('predictions.csv', 'cohort_metrics.csv', 'macro_metrics.csv',
                     'bootstrap_indices.npz', 'bootstrap_index_metadata.json', 'bootstrap_protocol.json')]
    source_files += [LOCO / 'mmr4_loco_predictions.csv', LOCO / 'tables/table_mmr4_loco_cutoffs.csv']
    hashes = {str(p): sha(p) for p in source_files}
    pred = pd.read_csv(SOURCE / 'predictions.csv', float_precision='round_trip')
    pred = pred[(pred.panel == 'panel12') & (pred.split != 'development_apparent')].copy()
    pred = pred.sort_values(['dataset', 'sampleID']).reset_index(drop=True)
    assert len(pred) == 1212 and not pred.duplicated(['dataset', 'sampleID']).any()
    mmr = pd.read_csv(LOCO / 'mmr4_loco_predictions.csv', float_precision='round_trip')
    paired = pred.merge(mmr[['dataset', 'sampleID', 'y', 'MMR4_LOCO']], on=['dataset', 'sampleID'],
                        validate='one_to_one', suffixes=('', '_locked'))
    assert len(paired) == len(mmr)
    np.testing.assert_array_equal(paired.y, paired.y_locked)
    np.testing.assert_allclose(paired.MMR4_LOCO, paired.MMR4_LOCO_locked, atol=2e-16, rtol=0)
    np.testing.assert_allclose(paired.Fusion05, .5 * paired.Subtype + .5 * paired.MMR4_LOCO, atol=2e-16, rtol=0)
    paired = paired.drop(columns=['y_locked', 'MMR4_LOCO_locked'])
    metric = pd.read_csv(SOURCE / 'cohort_metrics.csv', float_precision='round_trip')
    metric = metric[(metric.panel == 'panel12') & (metric.split != 'development_apparent')]
    meta = {v['dataset']: v for v in json.loads((SOURCE / 'bootstrap_index_metadata.json').read_text())}
    arrays = np.load(SOURCE / 'bootstrap_indices.npz')
    # Reproduce the archived stream exactly, without inventing a new seed or
    # resetting the generator inside either the cohort or resampling loop.
    rng = np.random.default_rng(2026)
    rows, boot, indices_out = [], {}, {}
    for cohort, part in paired.groupby('dataset', sort=True):
        y = part.y.to_numpy(dtype=int)
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        assert part.sampleID.tolist() == meta[cohort]['sample_order']
        index = arrays[cohort]
        assert index.shape == (2000, len(part))
        replay = np.concatenate([rng.choice(pos, (2000, len(pos)), replace=True),
                                 rng.choice(neg, (2000, len(neg)), replace=True)], axis=1)
        np.testing.assert_array_equal(index, replay)
        assert np.all(y[index[:, :len(pos)]] == 1) and np.all(y[index[:, len(pos):]] == 0)
        indices_out[cohort] = index
        record = {'dataset': cohort, 'split': part.split.iloc[0], 'n': len(part),
                  'MSI': len(pos), 'MSS': len(neg), 'MMR4_held_out_cohort': cohort}
        lock = json.loads((LOCO / 'locks' / ('heldout_' + cohort + '.json')).read_text())
        assert cohort not in lock['training_cohorts'] and not lock['held_out_labels_used_for_fit']
        record['MMR4_training_cohorts'] = ';'.join(lock['training_cohorts'])
        for column, label in [('Fusion05', 'Fusion'), ('MMR4_LOCO', 'MMR4')]:
            score = part[column].to_numpy()
            point = float(roc_auc_score(y, score))
            saved = metric[(metric.dataset == cohort) & (metric.component == column)].AUROC.iloc[0]
            assert abs(point - saved) < 2e-15
            ranks = rankdata(score[index], axis=1, method='average')
            values = (ranks[:, :len(pos)].sum(axis=1) - len(pos) * (len(pos)+1)/2) / (len(pos)*len(neg))
            np.testing.assert_allclose(values, weighted_pair_auc(y, score, index), rtol=0, atol=2e-15)
            boot[(cohort, label)] = values
            record[label + '_AUROC'] = point
        delta = boot[(cohort, 'Fusion')] - boot[(cohort, 'MMR4')]
        boot[(cohort, 'Delta')] = delta
        record['Delta_AUROC_Fusion_minus_MMR4'] = record['Fusion_AUROC'] - record['MMR4_AUROC']
        record['CI_low'], record['CI_high'] = np.quantile(delta, [.025, .975]).tolist()
        record['bootstrap_resamples'] = 2000
        rows.append(record)
    cohort_table = pd.DataFrame(rows)
    macro_rows = []
    scopes = [('development_OOF', DEV), ('retrospective', sorted(set(paired.dataset) - set(DEV))),
              ('all_eight', sorted(paired.dataset.unique()))]
    for scope, cohorts in scopes:
        part = cohort_table[cohort_table.dataset.isin(cohorts)]
        values = np.mean([boot[(c, 'Delta')] for c in cohorts], axis=0)
        record = {'scope': scope, 'n_cohorts': len(cohorts), 'cohorts': ';'.join(cohorts),
                  'n_samples': int(part.n.sum()), 'Fusion_macro_AUROC': float(part.Fusion_AUROC.mean()),
                  'MMR4_macro_AUROC': float(part.MMR4_AUROC.mean()),
                  'Delta_AUROC_Fusion_minus_MMR4': float(part.Delta_AUROC_Fusion_minus_MMR4.mean()),
                  'CI_low': float(np.quantile(values, .025)), 'CI_high': float(np.quantile(values, .975)),
                  'bootstrap_resamples': 2000}
        assert abs(record['Delta_AUROC_Fusion_minus_MMR4'] - (record['Fusion_macro_AUROC'] - record['MMR4_macro_AUROC'])) < 2e-15
        macro_rows.append(record)
        boot[(scope, 'macro_Delta')] = values
    macro = pd.DataFrame(macro_rows)
    for frame, name in [(cohort_table, 'S13_fusion_vs_MMR4_cohort.csv'), (macro, 'S13_fusion_vs_MMR4_macro.csv'),
                        (paired, 'paired_sample_predictions.csv')]:
        frame.to_csv(OUT / name, index=False, float_format='%.17g')
    np.savez_compressed(OUT / 'bootstrap_indices.npz', **indices_out)
    np.savez_compressed(OUT / 'bootstrap_AUROC_distributions.npz', **{'__'.join(k): v for k, v in boot.items()})
    shutil.copy2(SOURCE / 'bootstrap_index_metadata.json', OUT / 'bootstrap_index_metadata.json')
    protocol = {'seed': 2026, 'resamples': 2000, 'prediction_identity': 'actual panel12 five-fold OOF/retrospective; archived eight-round LOCO MMR4',
                'comparison': 'AUROC(Fusion05) minus AUROC(MMR4_LOCO)', 'interval': 'paired outcome-stratified sample bootstrap; percentile95',
                'resampling_unit': 'bulk sample within cohort; TCGA patient IDs distinct; GEO patient clustering not independently audited',
                'macro': 'equal cohort weight; same replicate combines independent within-cohort resamples; no cohort resampling',
                'draw_identity': 'exact archived RNG2026 stream, independently reproduced; no seed offsets or loop reset',
                'conditional_on': 'fixed panel, features, classifier predictions, cohort Z-score and LOCO cutoff locks',
                'models_refitted': False, 'external_outcomes_used_for_selection': False, 'source_sha256': hashes}
    dump(OUT / 'protocol.json', protocol)
    assert hashes == {str(p): sha(p) for p in source_files}
    result = {'status': 'passed', 'n_paired_samples': len(paired), 'cohorts_checked': len(rows),
              'paired_alignment_checked': True, 'archived_rng_stream_reproduced': True,
              'all_32000_bootstrap_component_AUROCs_verified_with_independent_weighted_pair_method': True,
              'macro_results': macro_rows, 'outputs_sha256': {f.name: sha(f) for f in OUT.iterdir() if f.is_file()}}
    dump(OUT / 'verification.json', result)
    print(cohort_table[['dataset', 'Delta_AUROC_Fusion_minus_MMR4', 'CI_low', 'CI_high']].to_string(index=False))
    print(macro.to_string(index=False))

if __name__ == '__main__': main()
