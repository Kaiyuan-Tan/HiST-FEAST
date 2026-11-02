import contextlib
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch import nn

import train
import train_patch


class TinyUNI(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.batch_sizes = []
        self.grad_enabled = []

    def forward(self, images):
        self.batch_sizes.append(len(images))
        self.grad_enabled.append(torch.is_grad_enabled())
        return images.float().mean(dim=(2, 3)) / 255 + self.offset


class TinyGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.BatchNorm1d(3)
        self.dropout = nn.Dropout(0.5)
        self.graph_sizes = []

    def forward(self, features, adjacency):
        self.graph_sizes.append(len(features))
        return torch.sparse.mm(adjacency, self.dropout(self.norm(features)))


class TinyHead(nn.Module):
    def __init__(self, genes):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(1.0))
        self.genes = genes

    def forward(self, features):
        return features.new_ones((len(features), self.genes)) * self.bias


class TinyModel(nn.Module):
    def __init__(self, feature_dim=1024, genes=2):
        super().__init__()
        self.uni = TinyUNI()
        self.gcn = TinyGCN()
        self.resnet = TinyHead(genes)

    def forward(self, images, adjacency):
        return self.resnet(self.gcn(self.uni(images), adjacency))


def slide_batch(nodes, target):
    images = torch.full((nodes, 3, 4, 4), 64, dtype=torch.uint8)
    indices = torch.arange(nodes).repeat(2, 1)
    adjacency = torch.sparse_coo_tensor(indices, torch.ones(nodes), (nodes, nodes)).coalesce()
    return ((images, adjacency), torch.full((nodes, 2), target, dtype=torch.float32))


def make_slide(root, sample_id, genes):
    (root / "patches").mkdir(parents=True, exist_ok=True)
    (root / "st").mkdir(exist_ok=True)
    with h5py.File(root / "patches" / f"{sample_id}.h5", "w") as handle:
        handle.create_dataset("img", data=np.full((2, 224, 224, 3), 64, dtype=np.uint8))
        handle.create_dataset("barcodes", data=np.array([b"spot2", b"spot1"]))
        handle.create_dataset("coords", data=np.array([[0, 0], [20, 0]]))
    data = ad.AnnData(
        X=sp.csr_matrix(np.array([[2, 3, 4], [4, 5, 6]], dtype=np.uint16)),
        obs=pd.DataFrame(index=["spot1", "spot2"]), var=pd.DataFrame(index=genes),
    )
    data.write_h5ad(root / "st" / f"{sample_id}.h5ad")


class ValidationTests(unittest.TestCase):
    def quiet(self):
        stack = contextlib.ExitStack()
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
        return stack

    def validate(self, module, model, loader):
        args = (model, loader, torch.device("cpu"), 0, 1)
        return module.validate_one_epoch(*args, 2) if module is train_patch else module.validate_one_epoch(*args)

    def test_validation_is_eval_no_grad_and_means_per_slide_losses(self):
        for module in (train, train_patch):
            with self.subTest(module=module.__name__), self.quiet():
                model = TinyModel().train()
                before = {name: value.clone() for name, value in model.state_dict().items()}
                loader = [slide_batch(2, 2), slide_batch(5, 4)]
                self.assertAlmostEqual(self.validate(module, model, loader), 5.0)
                self.assertFalse(model.training)
                self.assertFalse(model.gcn.norm.training)
                self.assertTrue(all(not flag for flag in model.uni.grad_enabled))
                self.assertTrue(all(p.grad is None for p in model.parameters()))
                for name, value in model.state_dict().items():
                    self.assertTrue(torch.equal(value, before[name]), name)
                self.assertEqual(model.gcn.graph_sizes, [2, 5])
                self.assertEqual(model.uni.batch_sizes, [2, 2, 2, 1] if module is train_patch else [2, 5])

    def test_validation_rejects_empty_loader_bad_targets_and_shape(self):
        for module in (train, train_patch):
            with self.subTest(module=module.__name__), self.quiet():
                with self.assertRaisesRegex(ValueError, "No slides"):
                    self.validate(module, TinyModel(), [])
                with self.assertRaisesRegex(ValueError, "nonnegative"):
                    self.validate(module, TinyModel(), [slide_batch(2, -1)])
                with self.assertRaisesRegex(ValueError, "finite"):
                    self.validate(module, TinyModel(), [slide_batch(2, float("nan"))])
                with self.assertRaisesRegex(ValueError, "Prediction shape"):
                    self.validate(module, TinyModel(genes=1), [slide_batch(2, 1)])

    def test_gene_selection_uses_training_only_and_aligns_validation(self):
        with tempfile.TemporaryDirectory() as temp, self.quiet():
            root, val = Path(temp) / "train", Path(temp) / "val"
            make_slide(root, "train1", ["g2", "g1", "train_only"])
            make_slide(val, "val1", ["g1", "val_only", "g2"])
            dataset, val_dataset = train.prepare_datasets(root, 2, 550, val)
            self.assertEqual(dataset.gene_names, ["g2", "g1"])
            self.assertEqual(val_dataset.gene_names, dataset.gene_names)
            np.testing.assert_array_equal(val_dataset[0][1].numpy(), [[6, 4], [4, 2]])
            # Validation availability must not silently change the training target set.
            with self.assertRaisesRegex(ValueError, "train_only"):
                train.prepare_datasets(root, 3, 550, val)

    def test_overlapping_slides_and_empty_validation_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp, self.quiet():
            root, val = Path(temp) / "train", Path(temp) / "val"
            make_slide(root, "same_id", ["g2", "g1", "other"])
            make_slide(val, "same_id", ["g2", "g1", "other"])
            with self.assertRaisesRegex(ValueError, "overlap"):
                train.prepare_datasets(root, 2, 550, val)
            with self.assertRaisesRegex(ValueError, "No validation patch files"):
                train.prepare_datasets(root, 2, 550, Path(temp) / "empty")

    def test_best_checkpoint_tracks_validation_instead_of_last_epoch(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            model = TinyModel()
            optimizer = torch.optim.AdamW(model.parameters())
            for epoch, val_losses in ((1, [2.0]), (2, [2.0, 3.0]), (3, [2.0, 3.0, 1.0])):
                with torch.no_grad():
                    model.resnet.bias.fill_(epoch)
                train.save_epoch(output, model, optimizer, epoch, ["g2", "g1"], {},
                                 [float(i) for i in range(epoch)], val_losses)
                best = torch.load(output / "best.pt", weights_only=True)
                last = torch.load(output / "last.pt", weights_only=True)
                expected_best = 1 if epoch < 3 else 3
                self.assertEqual(best["epoch"], expected_best)
                self.assertEqual(best["resnet_state_dict"]["bias"].item(), expected_best)
                self.assertEqual(last["epoch"], epoch)
                self.assertEqual(last["resnet_state_dict"]["bias"].item(), epoch)
                self.assertEqual(last["best_val_mse"], min(val_losses))
            self.assertEqual(json.loads((output / "val_loss_history.json").read_text()), [2.0, 3.0, 1.0])

    def test_check_data_checks_both_sets_without_loading_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root, val = Path(temp) / "train", Path(temp) / "val"
            make_slide(root, "train1", ["g2", "g1", "train_only"])
            make_slide(val, "val1", ["g1", "val_only", "g2"])
            fake = types.ModuleType("model.mymodel")
            fake.MyModel = Mock(side_effect=AssertionError("Model must not load for --check-data"))
            for module in (train, train_patch):
                with self.subTest(module=module.__name__), self.quiet(), patch.dict("sys.modules", {"model.mymodel": fake}):
                    module.main(["--root", str(root), "--val-root", str(val), "--genes", "2", "--check-data"])
            fake.MyModel.assert_not_called()

    def test_both_entrypoints_train_and_validate_end_to_end(self):
        # Real HESTDataset I/O, optimizer steps and checkpoints; a tiny frozen UNI
        # replacement avoids requiring the large pretrained checkpoint in tests.
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root, val = base / "train", base / "val"
            make_slide(root, "train1", ["g2", "g1", "train_only"])
            make_slide(val, "val1", ["g1", "val_only", "g2"])
            (base / "ckpts").mkdir()
            (base / "ckpts" / "pytorch_model.bin").write_bytes(b"test placeholder")
            fake = types.ModuleType("model.mymodel")
            fake.MyModel = TinyModel
            for module in (train, train_patch):
                for with_val in (False, True):
                    output = base / f"{module.__name__}_{with_val}"
                    args = ["--root", str(root), "--output-dir", str(output),
                            "--genes", "2", "--epochs", "2", "--device", "cpu"]
                    if with_val:
                        args += ["--val-root", str(val)]
                    with self.subTest(module=module.__name__, val=with_val), self.quiet(), \
                         patch.object(module, "ROOT", base), patch.dict("sys.modules", {"model.mymodel": fake}):
                        module.main(args)
                    last = torch.load(output / "last.pt", weights_only=True)
                    self.assertEqual(len(last["loss_history"]), 2)
                    self.assertEqual(len(last["val_loss_history"]), 2 if with_val else 0)
                    self.assertEqual(last["settings"]["gene_names"], ["g2", "g1"])
                    self.assertEqual(last["settings"]["val_slides"], ["val1"] if with_val else [])
                    self.assertTrue(last["optimizer_state_dict"]["state"])
                    self.assertEqual((output / "best.pt").exists(), with_val)
                    self.assertEqual((output / "val_loss_history.json").exists(), with_val)
                    if with_val:
                        best = torch.load(output / "best.pt", weights_only=True)
                        self.assertEqual(best["best_val_mse"], min(last["val_loss_history"]))
                        self.assertEqual(best["epoch"], last["best_epoch"])


if __name__ == "__main__":
    unittest.main()
