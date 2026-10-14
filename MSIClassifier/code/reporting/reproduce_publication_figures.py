"""Plot current bulk/S1-S3 data; export other final SVG assets without fitting."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,shutil
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lxml import etree as E
import cairosvg

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'output/figures')
    args=parser.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    workspace=out/'workspace';workspace.mkdir(exist_ok=True)
    bulk=ROOT/'code/bulk';snapshot=bulk/'classifier_snapshot'
    shutil.copytree(bulk/'stability_archive',workspace/'stability',dirs_exist_ok=True)
    (workspace/'source_archive').mkdir(exist_ok=True)
    shutil.copy2(HERE/'resources/source_manuscript_bulk12.docx',workspace/'source_archive/source_manuscript.docx')
    plot_source=HERE/'historical_bulk12_build/build_tables_figures.py'
    spec=importlib.util.spec_from_file_location('retained_bulk_plotting',plot_source)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    mod.HERE=workspace;mod.FIG=out;mod.SUP=out/'data';mod.SUP.mkdir(exist_ok=True)
    metrics=pd.read_csv(snapshot/'cohort_metrics.csv',float_precision='round_trip');metrics=metrics[metrics.panel=='panel12'].copy()
    macro=pd.read_csv(snapshot/'macro_metrics.csv',float_precision='round_trip');macro=macro[macro.panel=='panel12'].copy()
    stability=pd.read_csv(bulk/'stability_archive/S14.csv',float_precision='round_trip').set_index('feature')
    summary=json.loads((bulk/'stability_archive/summary.json').read_text())
    assert len(stability)==17 and stability.retained_bulk12.sum()==12
    # Compare numeric sources against the actual English supplementary table,
    # independent of the plotting function's own formatting.
    current=pd.read_csv(ROOT/'supplement/data_attachments/S13.csv',float_precision='round_trip')
    columns=['panel','split','dataset','component','AUROC','BA','sensitivity','specificity','Brier']
    pd.testing.assert_frame_equal(metrics[columns].sort_values(columns[:4]).reset_index(drop=True),current[columns].sort_values(columns[:4]).reset_index(drop=True),check_exact=False,atol=1e-12,rtol=0)
    mod.figures(metrics,macro,stability,summary)
    mod.edit_vector_figures(metrics)
    # Reuse the exact original uncapped sensitivity plotting block.
    tables=ROOT/'code/scRNA/project/最终模型/report/article/scRNA_AUROC_threshold_all_candidates_seed2026_20261013/tables'
    threshold=pd.read_csv(tables/'S10_threshold_performance.csv',float_precision='round_trip')
    screen=threshold[threshold.family.eq('subtype_threshold')].copy()
    fixed18=threshold[threshold.family.eq('fixed_panel_reference')].copy()
    cnv=pd.read_csv(tables/'S3_CNV_quantile_performance.csv',float_precision='round_trip')
    block=HERE/'resources/plot_scRNA_sensitivity_block.py'
    scope={'plt':plt,'np':np,'pd':pd,'shutil':shutil,'screen':screen,'fixed18':fixed18,'cnv':cnv,'FIG':out,'SUP':mod.SUP}
    exec(compile(block.read_text(),str(block),'exec'),scope)
    manifest=json.loads((HERE/'historical_manuscript_build/figure_label_manifest.json').read_text())
    svgns={'s':'http://www.w3.org/2000/svg'}
    edits=[]
    for record in manifest['files']:
        filename=Path(record['output_svg']).name
        p=out/filename
        if not p.exists():
            # Remaining current SC panels are editable SVG assets, not newly
            # regenerated measurements or newly fitted model figures.
            shutil.copy2(HERE/'publication_figures'/filename,p)
            edits.append({'figure':filename,'mode':'retained final SVG export'})
        else:
            root=E.fromstring(p.read_bytes());found=[]
            for change in record['changes']:
                matches=[t for t in root.xpath('//s:text',namespaces=svgns) if ''.join(t.itertext())==change['old']]
                if not matches:
                    assert any(''.join(t.itertext())==change['new'] for t in root.xpath('//s:text',namespaces=svgns)),change
                    continue
                for t in matches:
                    assert len(t)==0
                    t.text=change['new']
                    for k,v in change.get('text_attributes_after',{}).items():t.set(k,v)
                found.append({'old':change['old'],'new':change['new'],'matches':len(matches)})
            p.write_bytes(E.tostring(root,xml_declaration=True,encoding='UTF-8'))
            edits.append({'figure':filename,'mode':'numerical plot plus audited labels' if filename.startswith('Figure4') else 'retained source vector updated with current numeric cells/labels','edits':found})
        dims=record['output_png_dimensions']
        cairosvg.svg2png(url=str(p),write_to=str(p.with_suffix('.png')),output_width=dims[0],output_height=dims[1])
        cairosvg.svg2pdf(url=str(p),write_to=str(p.with_suffix('.pdf')))
    for source in (HERE/'publication_figures').glob('*.svg'):
        if (out/source.name).exists():continue
        dest=out/source.name;shutil.copy2(source,dest)
        cairosvg.svg2png(url=str(dest),write_to=str(dest.with_suffix('.png')))
        cairosvg.svg2pdf(url=str(dest),write_to=str(dest.with_suffix('.pdf')))
        edits.append({'figure':source.name,'mode':'retained final SVG export'})
    # Current spatial image is the paired source reproduction in code/spatial.
    shutil.copy2(HERE/'publication_figures/Figure3_spatial_transfer_unchanged.png',out/'Figure3_spatial_transfer_unchanged.png')
    inputs=[plot_source,block,snapshot/'cohort_metrics.csv',snapshot/'macro_metrics.csv',bulk/'stability_archive/S14.csv',bulk/'stability_archive/summary.json',tables/'S10_threshold_performance.csv',tables/'S3_CNV_quantile_performance.csv']
    report={'status':'passed','numeric_sources_match_current_S13':True,'current_panel12_n':12,'S2_all_candidates_n':17,'S3_uncapped_candidate_counts':[39,24,17,9],'model_fit_performed':False,'input_hashes':{str(p.relative_to(ROOT)):sha(p) for p in inputs},'figure_modes':edits,'outputs':[p.name for p in sorted(out.glob('*')) if p.suffix in ['.svg','.png','.pdf']]}
    (out/'figure_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('PASS: regenerated bulk/S1-S3 plots and exported current vectors;',len(report['outputs']),'figure files')

if __name__=='__main__':main()
