"""Numerical contracts for expression inputs, independent of model fitting."""
from pathlib import Path
import importlib.util
import tempfile
import unittest
import numpy as np
import pandas as pd
from scipy import sparse
import anndata as ad

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "workflows/expression/inputs.py"


class ExpressionContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if MODULE.exists():
            spec = importlib.util.spec_from_file_location("expression_inputs", MODULE)
            cls.m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.m)

    def setUp(self):
        self.assertTrue(MODULE.exists(), "Raw-expression input module is missing")

    def test_bulk_scales_and_rounding(self):
        original = pd.DataFrame([[0., 3.], [7., 15.]], index=["MLH1", "MSH2"], columns=["A", "B"])
        linear, sampling, log_values = self.m.bulk_scales(np.log2(original + 1), "log2p1")
        np.testing.assert_allclose(linear, original)
        np.testing.assert_array_equal(sampling, original)
        np.testing.assert_allclose(log_values, np.log2(original + 1))
        with self.assertRaisesRegex(ValueError, "negative"):
            self.m.bulk_scales(original - 1, "linear")

    def test_duplicate_identifiers_are_rejected_before_pandas_suffixing(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "bulk.csv"
            f.write_text("gene,A,A\nMLH1,1,2\n")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                self.m.read_bulk(f)

    def test_probe_mapping_is_explicit_and_median_in_supplied_scale(self):
        x = pd.DataFrame([[2., 4.], [6., 8.], [1., 3.]], index=["p1", "p2", "p3"], columns=["A", "B"])
        mp = pd.DataFrame({"probe_id": ["p1", "p2", "p3"], "gene_symbol": ["MLH1", "MLH1", "MSH2"]})
        genes = self.m.map_probes(x, mp)
        np.testing.assert_allclose(genes.loc["MLH1"], [4, 6])
        bad = pd.concat([mp, pd.DataFrame({"probe_id": ["p1"], "gene_symbol": ["PMS2"]})])
        with self.assertRaisesRegex(ValueError, "one gene"):
            self.m.map_probes(x, bad)

    def test_count_normalization_uses_full_library(self):
        a = ad.AnnData(sparse.csr_matrix([[1., 3.], [0., 4.]]))
        a.var_names = ["MLH1", "OTHER"]
        out = self.m.log_expression(a, "counts")
        np.testing.assert_allclose(out.X.toarray(), np.log1p([[2500., 7500.], [0., 10000.]]))
        np.testing.assert_array_equal(a.X.toarray(), [[1, 3], [0, 4]])
        with self.assertRaisesRegex(ValueError, "integer"):
            self.m.log_expression(ad.AnnData(np.array([[.5, .5]])), "counts")

    def test_parent_denominator_includes_unknown_subtype_and_missing_parent(self):
        cells = pd.DataFrame({"sampleID": ["A"] * 3 + ["B"],
                              "major": ["TNKILC", "TNKILC", "Epi", "Epi"],
                              "subtype": ["cTNI04", "", "", ""]})
        r = self.m.cell_fractions(cells, ["cTNI04"], {"cTNI04"}).set_index("sampleID")
        self.assertEqual(r.loc["A", "cTNI04"], .5)
        self.assertTrue(np.isnan(r.loc["B", "cTNI04"]))
        with self.assertRaisesRegex(ValueError, "unsupported"):
            self.m.cell_fractions(cells, ["cTNI04"], set())

    def test_weighting_uses_same_retained_cnv_cells_and_preserves_zero(self):
        cells = pd.DataFrame({"sampleID": ["A", "A", "A", "B", "C"],
                              "is_epithelial": [True, True, False, True, True],
                              "candidate": [True, False, False, True, False],
                              "MLH1": [2., 8., 100., 0., 5.]})
        r = self.m.weighted_mlh1(cells).set_index("sampleID")
        self.assertEqual(r.loc["A", "weighted_MLH1"], 1.)
        self.assertEqual(r.loc["A", "n_all_epithelial"], 2)
        self.assertEqual(r.loc["B", "weighted_MLH1"], 0.)
        self.assertTrue(np.isnan(r.loc["C", "weighted_MLH1"]))

    def test_spatial_proxy_uses_all_shared_genes(self):
        allocated = sparse.csr_matrix([[2., 8.], [0., 4.]])
        abund = pd.DataFrame({"Epi_CNV_candidate": [1., 3.], "Epi_other": [1., 1.]})
        result = self.m.spatial_weighted_mlh1(allocated, 0, abund)
        expected = np.log1p(2000.) / 6.
        self.assertAlmostEqual(result["weighted_MLH1"], expected)
        self.assertEqual(result["n_genes_normalization"], 2)

    def test_h5ad_and_transposed_text_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = ad.AnnData(sparse.csr_matrix([[1, 2], [3, 4]], dtype=float))
            a.obs_names = ["cell_A", "cell_B"]; a.var_names = ["MLH1", "MSH2"]
            a.write_h5ad(Path(tmp) / "a.h5ad")
            got = self.m.read_cells(Path(tmp) / "a.h5ad")
            self.assertEqual(got.obs_names.tolist(), a.obs_names.tolist())
            f = Path(tmp) / "a.csv"
            pd.DataFrame(a.X.toarray(), index=a.obs_names, columns=a.var_names).T.to_csv(f)
            got = self.m.read_cells(f, transpose=True)
            np.testing.assert_array_equal(got.X.toarray(), a.X.toarray())


if __name__ == "__main__":
    unittest.main()
