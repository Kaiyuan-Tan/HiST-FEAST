import contextlib
import io
import tempfile
import unittest
import warnings
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from torchvision.transforms import PILToTensor

from dataloader.HESTDataset import HESTDataset, read_gene_expression
from train import check_dataset, select_gene_names


class DuplicateGeneTests(unittest.TestCase):
    def setUp(self):
        self.warning_context = warnings.catch_warnings()
        self.warning_context.__enter__()
        warnings.filterwarnings("ignore", message="Variable names are not unique", category=UserWarning)
        self.counts = np.array([[60000, 3, 10000, 5, 6], [8, 2, 9, 1, 2]], dtype=np.uint16)

    def tearDown(self):
        self.warning_context.__exit__(None, None, None)

    def make_data(self, matrix=None):
        return ad.AnnData(
            X=self.counts if matrix is None else matrix,
            obs=pd.DataFrame(index=["spot1", "spot2"]),
            var=pd.DataFrame(index=["A", "B", "A", "BLANK_1", "BLANK_1"]),
        )

    def test_sum_duplicates_align_barcodes_and_gene_order_without_uint16_overflow(self):
        for matrix in (self.counts, sp.csr_matrix(self.counts), sp.csc_matrix(self.counts)):
            with self.subTest(matrix=type(matrix).__name__):
                data = self.make_data(matrix)
                result = read_gene_expression(data, ["spot2", "spot1"], ["B", "A"])
                np.testing.assert_array_equal(result, [[2, 17], [3, 70000]])
                self.assertEqual(result.dtype, np.float32)
                self.assertEqual(list(data.var_names), ["A", "B", "A", "BLANK_1", "BLANK_1"])

    def test_unique_names_keep_existing_expression(self):
        data = self.make_data()[:, [0, 1]].copy()
        result = read_gene_expression(data, ["spot2", "spot1"], ["B", "A"])
        np.testing.assert_array_equal(result, [[2, 8], [3, 60000]])

    def test_missing_gene_and_barcode_fail(self):
        data = self.make_data()
        with self.assertRaisesRegex(ValueError, "Missing target genes"):
            read_gene_expression(data, ["spot1"], ["absent"])
        with self.assertRaisesRegex(ValueError, "barcodes are missing"):
            read_gene_expression(data, ["absent"], ["A"])
        data.obs_names = ["spot1", "spot1"]
        with self.assertRaisesRegex(ValueError, "duplicate barcodes"):
            read_gene_expression(data, ["spot1"], ["A"])

    def test_backed_dense_and_sparse_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            for i, matrix in enumerate((self.counts, sp.csr_matrix(self.counts), sp.csc_matrix(self.counts))):
                path = Path(temp) / f"sample{i}.h5ad"
                self.make_data(matrix).write_h5ad(path)
                data = ad.read_h5ad(path, backed="r")
                try:
                    result = read_gene_expression(data, ["spot2", "spot1"], ["B", "A"])
                    np.testing.assert_array_equal(result, [[2, 17], [3, 70000]])
                finally:
                    data.file.close()

    def test_selection_preflight_and_dataset_use_same_unique_targets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "patches").mkdir()
            (root / "st").mkdir()
            self.make_data(sp.csr_matrix(self.counts)).write_h5ad(root / "st" / "sample1.h5ad")
            ad.AnnData(
                X=sp.csr_matrix(np.array([[4, 11, 12], [5, 13, 14]], dtype=np.uint16)),
                obs=pd.DataFrame(index=["spot1", "spot2"]),
                var=pd.DataFrame(index=["B", "A", "A"]),
            ).write_h5ad(root / "st" / "sample2.h5ad")
            for name in ("sample1", "sample2"):
                with h5py.File(root / "patches" / f"{name}.h5", "w") as handle:
                    handle.create_dataset("img", data=np.zeros((2, 224, 224, 3), dtype=np.uint8))
                    handle.create_dataset("barcodes", data=np.array([b"spot2", b"spot1"]))
                    handle.create_dataset("coords", data=np.array([[0, 0], [1, 0]]))
            with contextlib.redirect_stdout(io.StringIO()):
                genes = select_gene_names(root, 2)
                self.assertEqual(genes, ["A", "B"])
                with self.assertRaisesRegex(ValueError, "only 2 targets are shared"):
                    select_gene_names(root, 3)
                dataset = HESTDataset(root, genes, PILToTensor())
                check_dataset(dataset)
            (images, adjacency), labels = dataset[0]
            self.assertEqual(tuple(images.shape), (2, 3, 224, 224))
            self.assertEqual(tuple(adjacency.shape), (2, 2))
            np.testing.assert_array_equal(labels.numpy(), [[17, 2], [70000, 3]])
            np.testing.assert_array_equal(dataset[1][1].numpy(), [[27, 5], [23, 4]])
            # The source file keeps its original five columns after reading.
            unchanged = ad.read_h5ad(root / "st" / "sample1.h5ad", backed="r")
            try:
                self.assertEqual(unchanged.n_vars, 5)
                self.assertFalse(unchanged.var_names.is_unique)
            finally:
                unchanged.file.close()


if __name__ == "__main__":
    unittest.main()
