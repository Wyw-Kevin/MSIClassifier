"""Generate current AUROC-cutoff and17-subtype figures from numeric inputs."""
from pathlib import Path
import runpy
if __name__=='__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[1]/'scRNA/primary17/code/build_figures.py'),run_name='__main__')
