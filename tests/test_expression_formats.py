"""Verify 10x HDF5 and MTX structures through the actual Scanpy readers."""
from pathlib import Path
import gzip
import importlib.util
import tempfile
import unittest
import h5py
import numpy as np
from scipy import sparse, io

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("expression_inputs", ROOT / "workflows/expression/inputs.py")
inputs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inputs)


class FormatReaders(unittest.TestCase):
    def test_10x_hdf5(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "filtered_feature_bc_matrix.h5"
            x = sparse.csc_matrix([[1, 3], [2, 4]])
            with h5py.File(path, "w") as handle:
                matrix = handle.create_group("matrix")
                for name, values in [("data", x.data), ("indices", x.indices), ("indptr", x.indptr), ("shape", x.shape)]:
                    matrix.create_dataset(name, data=values)
                matrix.create_dataset("barcodes", data=np.array(["cellA", "cellB"], dtype="S"))
                features = matrix.create_group("features")
                features.create_dataset("id", data=np.array(["ENSG1", "ENSG2"], dtype="S"))
                features.create_dataset("name", data=np.array(["MLH1", "MSH2"], dtype="S"))
                features.create_dataset("feature_type", data=np.array(["Gene Expression"] * 2, dtype="S"))
                features.create_dataset("genome", data=np.array(["GRCh38"] * 2, dtype="S"))
            got = inputs.read_cells(path)
            self.assertEqual(got.obs_names.tolist(), ["cellA", "cellB"])
            self.assertEqual(got.var_names.tolist(), ["MLH1", "MSH2"])
            np.testing.assert_array_equal(got.X.toarray(), [[1, 2], [3, 4]])

    def test_10x_mtx_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            with gzip.open(path / "matrix.mtx.gz", "wb") as handle:
                io.mmwrite(handle, sparse.coo_matrix([[1, 3], [2, 4]]))
            with gzip.open(path / "features.tsv.gz", "wt") as handle:
                handle.write("ENSG1\tMLH1\tGene Expression\nENSG2\tMSH2\tGene Expression\n")
            with gzip.open(path / "barcodes.tsv.gz", "wt") as handle:
                handle.write("cellA\ncellB\n")
            got = inputs.read_cells(path)
            self.assertEqual(got.var_names.tolist(), ["MLH1", "MSH2"])
            np.testing.assert_array_equal(got.X.toarray(), [[1, 2], [3, 4]])


if __name__ == "__main__":
    unittest.main()
