from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
from scipy import sparse
import anndata as ad

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "predict.py"


class RawCLI(unittest.TestCase):
    def setUp(self):
        self.assertTrue(ENTRY.exists(), "Expression-to-score entry point is missing")

    def run_entry(self, *args):
        return subprocess.run([sys.executable, str(ENTRY), *map(str, args)], capture_output=True, text=True)

    def test_bulk_starts_with_gene_expression(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pd.DataFrame([[1, 2], [3, 4]], index=["MLH1", "MSH2"], columns=["A", "B"]).to_csv(tmp / "bulk.csv")
            out = tmp / "out"
            r = self.run_entry("bulk", "--expression", tmp / "bulk.csv", "--scale", "linear", "--dataset", "NEW",
                               "--output-dir", out, "--check-only")
            self.assertEqual(r.returncode, 0, r.stderr)
            report = json.loads((out / "input_check.json").read_text())
            self.assertEqual(report["n_samples"], 2)
            self.assertEqual(report["expression_scale"], "linear")
            self.assertFalse(report["model_executed"])

    def test_single_cell_h5ad_needs_sample_information(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            a = ad.AnnData(sparse.csr_matrix(np.ones((3, 2))))
            a.obs_names = ["c1", "c2", "c3"]; a.var_names = ["MLH1", "OTHER"]
            a.obs["Sample"] = "sampleA"; a.obs["Class"] = "Tumor"
            a.write_h5ad(tmp / "cells.h5ad")
            r = self.run_entry("scrna", "--expression", tmp / "cells.h5ad", "--scale", "counts", "--dataset", "NEW",
                               "--output-dir", tmp / "out", "--check-only")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(json.loads((tmp / "out/input_check.json").read_text())["n_cells"], 3)
            del a.obs["Sample"]; a.write_h5ad(tmp / "missing.h5ad")
            r = self.run_entry("scrna", "--expression", tmp / "missing.h5ad", "--scale", "counts", "--dataset", "NEW",
                               "--output-dir", tmp / "out2", "--check-only")
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("Sample", r.stderr)

    def test_spatial_rejects_log_values_as_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            a = ad.AnnData(np.array([[.5, 1.5], [.3, 1.3]]))
            a.var_names = ["MLH1", "OTHER"]; a.write_h5ad(tmp / "a.h5ad")
            r = self.run_entry("spatial", "--expression", tmp / "a.h5ad", "--sample-id", "sectionA",
                               "--output-dir", tmp / "out", "--check-only")
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("integer", r.stderr)


if __name__ == "__main__":
    unittest.main()
