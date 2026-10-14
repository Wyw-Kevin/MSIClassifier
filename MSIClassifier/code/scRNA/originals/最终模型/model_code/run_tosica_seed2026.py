"""Rebuild the hierarchical TOSICA checkpoints with one seed: 2026."""

import importlib.util
import sys
from pathlib import Path

ROOT = Path("/home/wyw/rstudio/MSI")
SOURCE = ROOT / "最终模型/Migrate/train_hierarchical_tosica.py"
OUT = ROOT / "最终模型/scRNA_model/reference_seed2026/Migrate"


def main():
    spec = importlib.util.spec_from_file_location("tosica_seed2026_source", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    module.OUT_DIR = OUT
    module.RANDOM_STATE = 2026
    original_sample_rows = module.sample_rows

    def sample_rows_seed2026(*args, **kwargs):
        kwargs["seed"] = 2026
        return original_sample_rows(*args, **kwargs)

    module.sample_rows = sample_rows_seed2026
    module.main()


if __name__ == "__main__":
    main()
