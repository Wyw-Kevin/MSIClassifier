"""Run retained numerical plotting functions, independent of Word documents."""
from pathlib import Path
import argparse,importlib.util,os,subprocess,sys,json,shutil
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',required=True,type=Path)
    args=p.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True);data=out/'data';data.mkdir(exist_ok=True)
    spec=importlib.util.spec_from_file_location('bulk_figures',HERE/'plot_bulk.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.FIG=out;m.SUP=data;bulk=HERE.parent/'bulk'
    metrics=pd.read_csv(bulk/'classifier_snapshot/cohort_metrics.csv',float_precision='round_trip');metrics=metrics[metrics.panel.eq('panel12')]
    macro=pd.read_csv(bulk/'classifier_snapshot/macro_metrics.csv',float_precision='round_trip');macro=macro[macro.panel.eq('panel12')]
    stability=pd.read_csv(bulk/'stability_archive/S14.csv',float_precision='round_trip').set_index('feature');summary=json.loads((bulk/'stability_archive/summary.json').read_text())
    m.figures(metrics,macro,stability,summary)
    tables=HERE.parent/'scRNA/resources/article/scRNA_AUROC_threshold_all_candidates_seed2026_20261013/tables'
    threshold=pd.read_csv(tables/'S10_threshold_performance.csv',float_precision='round_trip')
    namespace={'plt':plt,'np':np,'pd':pd,'shutil':shutil,'screen':threshold[threshold.family.eq('subtype_threshold')],'fixed18':threshold[threshold.family.eq('fixed_panel_reference')],'cnv':pd.read_csv(tables/'S3_CNV_quantile_performance.csv',float_precision='round_trip'),'FIG':out,'SUP':data}
    block=HERE/'plot_scRNA_sensitivity_block.py';exec(compile(block.read_text(),str(block),'exec'),namespace)
    env=os.environ.copy();env['MSI_FIGURE_OUTPUT']=str(out)
    subprocess.run([sys.executable,str(HERE/'plot_scRNA.py')],env=env,check=True)
    print('Numeric plots written to',out)

if __name__=='__main__':main()
