"""Recreate current17 article figures and unchanged bulk plots from numeric inputs."""
from pathlib import Path
import argparse,importlib.util,subprocess,sys,shutil,json
import pandas as pd
HERE=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',required=True,type=Path);args=p.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    snapshot=HERE.parent/'scRNA/primary17'
    for script in ['build_figures.py','build_spatial_figure.py']:subprocess.run([sys.executable,str(snapshot/'code'/script)],check=True)
    for f in (snapshot/'figures').iterdir():shutil.copy2(f,out/f.name)
    spec=importlib.util.spec_from_file_location('bulk_plots',HERE/'plot_bulk.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.FIG=out;module.SUP=out/'data';module.SUP.mkdir(exist_ok=True)
    bulk=HERE.parent/'bulk';metrics=pd.read_csv(bulk/'classifier_snapshot/cohort_metrics.csv');macro=pd.read_csv(bulk/'classifier_snapshot/macro_metrics.csv')
    stability=pd.read_csv(bulk/'stability_archive/S14.csv').set_index('feature');summary=json.loads((bulk/'stability_archive/summary.json').read_text())
    module.figures(metrics[metrics.panel.eq('panel12')],macro[macro.panel.eq('panel12')],stability,summary)
if __name__=='__main__':main()
