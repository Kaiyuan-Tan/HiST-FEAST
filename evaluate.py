"""Evaluate a saved HiST-FEAST checkpoint on held-out HEST slides.

python evaluate.py --root hest_data_splits/mouse/test --checkpoint outputs/train_patch_mouse/best.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import math

import h5py
import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from torchvision.transforms import PILToTensor
from tqdm.auto import tqdm

from dataloader.HESTDataset import HESTDataset, make_dataloader
from utils.checkpoints import load_checkpoint
from utils.data import check_dataset
from utils.features import extract_uni_features
from utils.load_model import build_model, resolve_device
from utils.paths import ROOT, project_path


def prepare_test_dataset(root, checkpoint):
    settings = checkpoint["settings"]
    dataset = HESTDataset(root=root, gene_names=checkpoint["gene_names"],
                          transform=PILToTensor(), radius=settings["radius"])
    if not len(dataset):
        raise ValueError(f"No test patch files found in {root / 'patches'}")
    check_dataset(dataset)
    return dataset


SUPPORTED_GENE_METRICS = ("R2", "PCC")


def validate_gene_metrics(metrics=None):
    requested = list(SUPPORTED_GENE_METRICS if metrics is None else metrics)
    unsupported = sorted(set(requested) - set(SUPPORTED_GENE_METRICS))
    if unsupported:
        raise ValueError(
            f"Unsupported gene-prediction metrics: {unsupported}. "
            f"Supported metrics: {list(SUPPORTED_GENE_METRICS)}"
        )
    return requested


def compute_gene_metrics(true, pred, metrics=None):
    """Use the same metric calls as the HiST reference implementation."""
    requested = validate_gene_metrics(metrics)
    results = {}
    for metric in requested:
        if metric == "R2":
            results[metric] = float(r2_score(true, pred))
        elif metric == "PCC":
            results[metric] = float(pearsonr(true, pred)[0])
    return results


def regression_metrics(target, prediction):
    """Flatten the evaluation vector and make undefined values JSON-compatible."""
    target, prediction = target.ravel(), prediction.ravel()
    if len(target) < 2:
        return {"R2": None, "PCC": None}
    values = compute_gene_metrics(target, prediction)
    return {name: value if math.isfinite(value) else None for name, value in values.items()}


def format_metric(value):
    return "undefined" if value is None else f"{value:.6f}"


class TestMetrics:
    """HiST pooled and per-slide R2/PCC, plus supplemental per-gene metrics."""

    def __init__(self, gene_names):
        self.gene_names = list(gene_names)
        self.count = 0
        self.slides = []
        self.predictions = []
        self.targets = []

    def update(self, sample_id, prediction, target):
        prediction = np.array(prediction, dtype=np.float64, copy=True)
        target = np.array(target, dtype=np.float64, copy=True)
        if (prediction.ndim != 2 or prediction.shape != target.shape
                or prediction.shape[1] != len(self.gene_names) or not len(prediction)):
            raise ValueError("Predictions and targets must share nonempty [N, G] shape.")
        if not np.isfinite(prediction).all() or not np.isfinite(target).all():
            raise ValueError("Predictions and targets must be finite.")
        if (target < 0).any():
            raise ValueError("Targets must be nonnegative raw counts.")
        self.slides.append({"id": sample_id, "patches": len(prediction),
                            **regression_metrics(target, prediction)})
        self.predictions.append(prediction)
        self.targets.append(target)
        self.count += len(prediction)

    def results(self):
        if not self.count:
            raise ValueError("No test predictions collected.")
        # Follow HiST: concatenate slides, flatten, then call sklearn/scipy.
        all_predictions = np.vstack(self.predictions)
        all_targets = np.vstack(self.targets)
        spot_metrics = regression_metrics(all_targets.ravel(), all_predictions.ravel())
        gene_rows = []
        for i, gene in enumerate(self.gene_names):
            values = regression_metrics(all_targets[:, i], all_predictions[:, i])
            gene_rows.append({"gene": gene, **values})
        slide_avg, slide_std, slide_valid = {}, {}, {}
        for name in ("R2", "PCC"):
            values = [row[name] for row in self.slides]
            slide_valid[name] = sum(value is not None for value in values)
            # HiST uses np.mean/np.std rather than dropping NaNs. Represent an
            # undefined aggregate as JSON null and report how many slides are valid.
            defined = slide_valid[name] == len(values)
            slide_avg[name] = float(np.mean(values)) if defined else None
            slide_std[name] = float(np.std(values, ddof=0)) if defined else None
        summary = {
            "slides": len(self.slides), "patches": self.count, "genes": len(self.gene_names),
            "spot_metrics": spot_metrics, "slide_avg_metrics": slide_avg, "slide_std_metrics": slide_std,
            "valid_slide_metrics": slide_valid,
            "valid_gene_metrics": {name: sum(row[name] is not None for row in gene_rows)
                                   for name in ("R2", "PCC")},
            "metric_protocol": "HiST: linear expression; flattened spot-gene entries; equal-weight slide means",
            "metric_implementation": "sklearn.metrics.r2_score and scipy.stats.pearsonr",
            "metric_source": "https://github.com/wwyi1828/HiST/blob/main/tasks/gene_prediction/trainer.py",
            "undefined_policy": "JSON null / blank CSV; undefined slides propagate to slide mean and std",
        }
        return summary, gene_rows


@torch.no_grad()
def evaluate(model, dataset, device, batch_size, prediction_dir=None):
    model.eval()
    metrics = TestMetrics(dataset.gene_names)
    loader = make_dataloader(dataset, batch_size=1, shuffle=False, num_workers=0)
    if prediction_dir is not None:
        prediction_dir.mkdir(parents=True, exist_ok=True)
    progress = tqdm(loader, desc="Test")
    for index, ((patches, adjacency), labels) in enumerate(progress):
        features = extract_uni_features(model.uni, patches, device, batch_size)
        adjacency = adjacency.coalesce().to(device)
        predictions = model.resnet(model.gcn(features, adjacency)).cpu().numpy()
        targets = labels.numpy()
        sample_id = dataset.path[index].stem
        metrics.update(sample_id, predictions, targets)
        progress.set_postfix(slide=sample_id, R2=format_metric(metrics.slides[-1]["R2"]),
                             PCC=format_metric(metrics.slides[-1]["PCC"]))
        if prediction_dir is not None:
            with h5py.File(dataset.path[index], "r") as handle:
                key = "barcodes" if "barcodes" in handle else "barcode"
                barcodes = np.asarray(handle[key].asstr()[:].reshape(-1), dtype=str)
            np.savez_compressed(
                prediction_dir / f"{sample_id}.npz", prediction=predictions, target=targets,
                gene_names=np.asarray(dataset.gene_names), barcodes=barcodes,
            )
        del patches, features, adjacency, labels, predictions, targets
    return metrics


def write_results(output_dir, metrics, provenance):
    summary, gene_rows = metrics.results()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps({**provenance, **summary}, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    for name, rows, columns in (
        ("slide_metrics.csv", metrics.slides, ["id", "patches", "R2", "PCC"]),
        ("gene_metrics.csv", gene_rows, ["gene", "R2", "PCC"]),
    ):
        with (output_dir / name).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Test data containing patches/ and st/")
    parser.add_argument("--checkpoint", required=True, help="Saved best.pt or last.pt")
    parser.add_argument("--output-dir", default="outputs/test")
    parser.add_argument("--uni-batch-size", type=int, help="Override saved UNI batch size; fallback 64")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--check-data", action="store_true", help="Check checkpoint and test metadata only")
    parser.add_argument("--save-predictions", action="store_true", help="Save aligned per-slide prediction/target NPZ files")
    args = parser.parse_args(argv)
    if args.uni_batch_size is not None and args.uni_batch_size <= 0:
        parser.error("--uni-batch-size must be positive")
    root, checkpoint_path = project_path(args.root), project_path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path)
    dataset = prepare_test_dataset(root, checkpoint)
    settings = checkpoint["settings"]
    batch_size = args.uni_batch_size if args.uni_batch_size is not None else settings.get("uni_batch_size", 64)
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("Invalid UNI batch size in checkpoint; pass --uni-batch-size.")
    print(f"Checkpoint epoch={checkpoint.get('epoch')}, genes={dataset.num_genes}, "
          f"test_slides={len(dataset)}, radius={settings['radius']}, uni_batch_size={batch_size}", flush=True)
    if args.check_data:
        print("PASS: test data matches checkpoint genes and metadata.")
        return
    weights_path = ROOT / "ckpts" / "pytorch_model.bin"
    if not weights_path.is_file():
        raise FileNotFoundError(f"UNI weights required: {weights_path}")
    device = resolve_device(args.device)
    model = build_model(checkpoint, uni_weights=weights_path, device=device)
    epoch = checkpoint.get("epoch")
    del checkpoint
    output_dir = project_path(args.output_dir)
    metrics = evaluate(model, dataset, device, batch_size,
                       output_dir / "predictions" if args.save_predictions else None)
    summary = write_results(output_dir, metrics, {
        "checkpoint": str(checkpoint_path), "checkpoint_epoch": epoch, "test_root": str(root),
        "gene_names": dataset.gene_names, "test_slides": [path.stem for path in dataset.path],
        "radius": settings["radius"], "device": str(device), "uni_batch_size": batch_size,
        "label_transform": "raw", "adjacency": "unnormalized",
    })
    print("Test spot metrics (all flattened spot-gene entries):")
    for name in ("R2", "PCC"):
        print(f"  {name}: {format_metric(summary['spot_metrics'][name])}")
    print("Test slide metrics (unweighted mean +/- std):")
    for name in ("R2", "PCC"):
        print(f"  {name}: {format_metric(summary['slide_avg_metrics'][name])} +/- "
              f"{format_metric(summary['slide_std_metrics'][name])} "
              f"({summary['valid_slide_metrics'][name]}/{summary['slides']} defined)")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
