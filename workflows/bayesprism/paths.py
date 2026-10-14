"""Path-only adapters; estimator settings and source vendor remain unchanged."""
from pathlib import Path
import os
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("MSI_PROJECT_ROOT", str(PACKAGE_ROOT.parents[2]))).expanduser().resolve()
WORK_ROOT = Path(os.environ.get("MSI_BAYESPRISM_WORKDIR", str(PACKAGE_ROOT / "runs" / "rebuild"))).expanduser().resolve()
EXPORT_ROOT = Path(os.environ.get("MSI_BAYESPRISM_EXPORT_DIR", str(WORK_ROOT / "exports"))).expanduser().resolve()
CURRENT_COHORTS = ["TCGA-COADREAD", "GSE39582", "GSE24550", "GSE25071", "GSE27544", "GSE35896", "GSE39084", "GSE75316"]
HISTORICAL_COHORTS = CURRENT_COHORTS + ["GSE92921"]
def selected_cohorts():
    choice = os.environ.get("MSI_BAYESPRISM_COHORTS", "current8")
    result = CURRENT_COHORTS if choice == "current8" else HISTORICAL_COHORTS if choice == "historical9" else choice.split(",")
    if not result or len(set(result)) != len(result) or any(x not in HISTORICAL_COHORTS for x in result):
        raise ValueError("MSI_BAYESPRISM_COHORTS must be current8, historical9 or unique supported cohort IDs")
    return result

def reference_input(name):
    generated = WORK_ROOT / "reference" / name
    return generated if generated.exists() else PACKAGE_ROOT / "resources" / "reference" / name

def existing_export(name):
    generated = EXPORT_ROOT / name
    return generated if generated.exists() else PACKAGE_ROOT / "resources" / "historical_exports" / name
