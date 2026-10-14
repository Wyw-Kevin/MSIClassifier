"""Collect immutable sources for the local, unpublished MSIClassifier package."""
from pathlib import Path
import hashlib, json, shutil

PROJECT = Path(__file__).resolve().parents[4]
ARTICLE = PROJECT / '最终模型/report/article'
SOURCE = ARTICLE / 'method_evidence_update_20261014'
PACKAGE = PROJECT / 'MSIClassifier'
records = []

def copy(src, dst):
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    assert hashlib.sha256(dst.read_bytes()).hexdigest() == digest
    records.append({'source': str(src.relative_to(PROJECT)), 'package': str(dst.relative_to(PACKAGE)), 'sha256': digest, 'bytes': dst.stat().st_size})

def tree(src, dst, skip=()):
    for p in sorted(src.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or p.suffix == '.pyc': continue
        if p.name in skip: continue
        copy(p, dst / p.relative_to(src))

def main():
    bulk = PACKAGE / 'code/bulk'
    tree(SOURCE / 'model_computation', bulk / 'classifier_snapshot')
    tree(SOURCE / 'paired_increment', bulk / 'paired_increment')
    copy(SOURCE / 'compute_paired_increment.py', bulk / 'compute_paired_increment.py')
    old_loco = ARTICLE / 'bulk_mmr4_loco_fixed05_seed2026_20261012'
    for name in ['loco.py', 'test_loco.py', 'run.py', 'protocol.json', 'source_manifest.json', 'verification.json']:
        if (old_loco / name).exists(): copy(old_loco / name, bulk / 'mmr4' / name)
    tree(old_loco / 'locks', bulk / 'mmr4/locks')
    for name in ['resampling_records.npz', 'development_inputs.csv', 'candidate_directions.csv', 'verification.json', 'protocol.json']:
        copy(ARTICLE / 'bulk_selection_bootstrap_seed2026_20261012' / name, bulk / 'selection_bootstrap_source24' / name)
    stability = ARTICLE / 'bulk12_loco_manuscript_seed2026_20261013'
    copy(stability / 'compute_stability.py', bulk / 'source_archive/compute_stability.py')
    tree(stability / 'stability', bulk / 'stability_archive')
    text = (stability / 'compute_stability.py').read_text()
    text = text.replace("MODEL=ARTICLE/'bulk_7_vs12_mlp_mmr4_loco_seed2026_20261013'", "MODEL=HERE/'classifier_snapshot'")
    text = text.replace("OLD=ARTICLE/'bulk_selection_bootstrap_seed2026_20261012'", "OLD=HERE/'selection_bootstrap_source24'")
    text = text.replace("out=HERE/'stability';out.mkdir(exist_ok=True)", "import argparse\n    parser=argparse.ArgumentParser()\n    parser.add_argument('--output', type=Path, default=HERE/'outputs/selection_stability')\n    args=parser.parse_args()\n    out=args.output;out.mkdir(parents=True,exist_ok=True)")
    (bulk / 'compute_selection_stability.py').write_text(text)
    reporting = PACKAGE / 'code/reporting'
    for name in ['build_manuscript.py','build_supplement.py','prepare_text_changes.py','verify_delivery.py','text_changes.json','figure_label_manifest.json','source_inventory.json']:
        copy(SOURCE / name, reporting / 'historical_manuscript_build' / name)
    for name in ['build_tables_figures.py','build_manuscript.py']:
        copy(stability / name, reporting / 'historical_bulk12_build' / name)
    copy(ARTICLE / 'merged_scrna_figure_20260928/merge_figures.py', reporting / 'historical_scRNA_build/merge_figures.py')
    tree(SOURCE / 'figures', reporting / 'publication_figures')
    tree(SOURCE / 'source_archive', reporting / 'historical_manuscript_build/source_archive')
    for audit in ['audit_bayesprism','audit_mmr4','audit_scrna','audit_spatial']:
        tree(SOURCE / audit, PACKAGE / 'provenance/method_audits' / audit)
    for p in (SOURCE / 'supplement').iterdir():
        if not p.is_file(): continue
        if p.name == 'SupplementaryTablesS1-S14_English.xlsx': copy(p, PACKAGE / 'supplement' / p.name)
        elif p.name.startswith('SupplementaryFigure'): copy(p, PACKAGE / 'supplement/figures' / p.name)
        elif p.suffix in ['.csv','.npz','.json','.xml']: copy(p, PACKAGE / 'supplement/data_attachments' / p.name)
        elif p.name == 'SupplementaryMethods_English.txt': copy(p, PACKAGE / 'supplement/source_archive' / p.name)
    tree(SOURCE / 'model_archive', PACKAGE / 'models')
    # Preserve the earlier capped sensitivity implementation as historical source;
    # the uncapped analysis remains in code/scRNA, provided by the modality curator.
    prior = ARTICLE / 'scRNA_threshold_cnv_sensitivity_seed2026_20261013'
    for p in prior.rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts or p.suffix in ['.docx','.zip','.pyc']: continue
        copy(p, PACKAGE / 'code/scRNA_sensitivity/historical_capped_analysis' / p.relative_to(prior))
    for name in ['AGENTS.md','docs/superpowers/specs/locked-transcriptomic-workflow.md','docs/superpowers/specs/weighted-spatial-transfer.md']:
        copy(PROJECT / name, PACKAGE / 'docs/approved_protocol_archive' / name)
    (PACKAGE / 'provenance/root_source_manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    print('Verified root source copies:',len(records))

if __name__ == '__main__': main()
