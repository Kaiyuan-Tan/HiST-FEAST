"""
python train.py
python train.py --genes 100 --epochs 3
python train.py --check-data
python train.py --root hest_data_splits/mouse/train --val-root hest_data_splits/mouse/val

"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms import PILToTensor
from tqdm.auto import tqdm

from dataloader.HESTDataset import HESTDataset, make_dataloader

ROOT = Path(__file__).resolve().parent
CONTROL_PREFIXES = ("BLANK_", "NegControlCodeword_", "NegControlProbe_")


def project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def select_gene_names(root, count):
    patch_paths = sorted((root / "patches").glob("*.h5"))
    if not patch_paths:
        raise FileNotFoundError(f"No patch files found in {root / 'patches'}")
    gene_names = None
    for path in patch_paths:
        data = ad.read_h5ad(root / "st" / f"{path.stem}.h5ad", backed="r")
        try:
            if not data.obs_names.is_unique or not data.var_names.is_unique:
                raise ValueError(f"{path.stem}: duplicate barcodes or feature names.")
            names = [g for g in data.var_names if not g.startswith(CONTROL_PREFIXES)]
            available = set(names)
            gene_names = names if gene_names is None else [g for g in gene_names if g in available]
        finally:
            data.file.close()
    if len(gene_names) < count:
        raise ValueError(f"Requested {count} genes, but only {len(gene_names)} targets are shared.")
    return gene_names[:count]


def check_dataset(dataset):
    for patch_path, gt_path in zip(dataset.path, dataset.gt_path):
        data = ad.read_h5ad(gt_path, backed="r")
        try:
            if not data.obs_names.is_unique or not data.var_names.is_unique:
                raise ValueError(f"{patch_path.stem}: duplicate barcodes or feature names.")
            with h5py.File(patch_path, "r") as handle:
                key = "barcodes" if "barcodes" in handle else "barcode"
                barcodes = handle[key].asstr()[:].reshape(-1)
                coords = handle["coords"][:]
                expected_shape = (len(barcodes), 224, 224, 3)
                if (not len(barcodes) or handle["img"].shape != expected_shape
                        or handle["img"].dtype != np.uint8):
                    raise ValueError(f"{patch_path.stem}: expected nonempty uint8 N x 224 x 224 x 3 images.")
                if len(np.unique(barcodes)) != len(barcodes):
                    raise ValueError(f"{patch_path.stem}: duplicate patch barcodes.")
                if (coords.shape != (len(barcodes), 2) or not np.isfinite(coords).all()):
                    raise ValueError(f"{patch_path.stem}: invalid patch coordinates.")
                if (data.obs_names.get_indexer(barcodes) < 0).any():
                    raise ValueError(f"{patch_path.stem}: unmatched patch barcodes.")
                missing = set(dataset.gene_names).difference(data.var_names)
                if missing:
                    raise ValueError(f"{patch_path.stem}: missing target genes: {sorted(missing)}")
                print(f"{patch_path.stem}: patches={expected_shape}, "
                      f"labels=({len(barcodes)}, {dataset.num_genes})", flush=True)
        finally:
            data.file.close()


def prepare_datasets(root, genes, radius, val_root=None):
    # Target selection uses training slides only. Validation uses the same order.
    gene_names = select_gene_names(root, genes)
    dataset = HESTDataset(root=root, gene_names=gene_names,
                          transform=PILToTensor(), radius=radius)
    val_dataset = None
    if val_root is not None:
        val_dataset = HESTDataset(root=val_root, gene_names=gene_names,
                                  transform=PILToTensor(), radius=radius)
        if not len(val_dataset):
            raise ValueError(f"No validation patch files found in {val_root / 'patches'}")
        overlap = {path.stem for path in dataset.path} & {path.stem for path in val_dataset.path}
        if overlap:
            raise ValueError(f"Train and validation slides overlap: {sorted(overlap)}")
    print("Checking training data:", flush=True)
    check_dataset(dataset)
    if val_dataset is not None:
        print("Checking validation data:", flush=True)
        check_dataset(val_dataset)
    return dataset, val_dataset


def train_one_epoch(model, loader, optimizer, device, epoch, epochs):
    model.train()
    model.uni.eval()
    total_loss = 0.0
    slides = 0
    progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}")

    for (patches, adjacency), labels in progress:
        # Send the entire slide through the model in a single forward pass.
        if patches.dtype != torch.uint8:
            raise TypeError("Use PILToTensor(): UNI expects raw uint8 images before /255.")
        patches = patches.to(device)
        adjacency = adjacency.coalesce().to(device)
        labels = labels.to(device, dtype=torch.float32)
        if not torch.isfinite(labels).all() or (labels < 0).any():
            raise ValueError("Labels must be finite, nonnegative stored counts.")
        optimizer.zero_grad(set_to_none=True)

        predictions = model(patches, adjacency)
        if predictions.shape != labels.shape:
            raise ValueError(f"Prediction shape {predictions.shape} != label shape {labels.shape}")
        loss = F.mse_loss(predictions, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Training loss is NaN or Inf.")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all()
               for p in model.parameters() if p.requires_grad):
            raise FloatingPointError("Training gradients contain NaN or Inf.")
        optimizer.step()

        total_loss += loss.item()
        slides += 1
        progress.set_postfix(mse=f"{loss.item():.6f}")
        del patches, adjacency, labels, predictions, loss

    if slides == 0:
        raise ValueError("No slides in the training loader.")
    # Match the notebook: average one MSE per slide, rather than per-patch weighting.
    return total_loss / slides


@torch.no_grad()
def validate_one_epoch(model, loader, device, epoch, epochs):
    model.eval()
    total_loss = 0.0
    slides = 0
    progress = tqdm(loader, desc=f"Val {epoch + 1}/{epochs}")
    for (patches, adjacency), labels in progress:
        if patches.dtype != torch.uint8:
            raise TypeError("Use PILToTensor(): UNI expects raw uint8 images before /255.")
        patches = patches.to(device)
        adjacency = adjacency.coalesce().to(device)
        labels = labels.to(device, dtype=torch.float32)
        if not torch.isfinite(labels).all() or (labels < 0).any():
            raise ValueError("Labels must be finite, nonnegative stored counts.")
        predictions = model(patches, adjacency)
        if predictions.shape != labels.shape:
            raise ValueError(f"Prediction shape {predictions.shape} != label shape {labels.shape}")
        loss = F.mse_loss(predictions, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Validation loss is NaN or Inf.")
        total_loss += loss.item()
        slides += 1
        progress.set_postfix(mse=f"{loss.item():.6f}")
        del patches, adjacency, labels, predictions, loss
    if slides == 0:
        raise ValueError("No slides in the validation loader.")
    return total_loss / slides


def save_epoch(output_dir, model, optimizer, epoch, gene_names, settings,
               loss_history, val_loss_history):
    best_epoch = (min(range(len(val_loss_history)), key=val_loss_history.__getitem__) + 1
                  if val_loss_history else None)
    best_val_mse = val_loss_history[best_epoch - 1] if best_epoch is not None else None
    # UNI is frozen and continues to load from the original .bin when rebuilding MyModel.
    checkpoint = {"epoch": epoch, "gene_names": gene_names, "settings": settings,
                  "gcn_state_dict": model.gcn.state_dict(),
                  "resnet_state_dict": model.resnet.state_dict(),
                  "optimizer_state_dict": optimizer.state_dict(),
                  "loss_history": loss_history, "val_loss_history": val_loss_history,
                  "best_val_mse": best_val_mse, "best_epoch": best_epoch}
    torch.save(checkpoint, output_dir / "last.pt")
    if best_epoch == epoch:
        torch.save(checkpoint, output_dir / "best.pt")
    (output_dir / "loss_history.json").write_text(
        json.dumps(loss_history, indent=2, allow_nan=False), encoding="utf-8"
    )
    if val_loss_history:
        (output_dir / "val_loss_history.json").write_text(
            json.dumps(val_loss_history, indent=2, allow_nan=False), encoding="utf-8"
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="hest_data")
    parser.add_argument("--val-root", help="Validation directory containing patches/ and st/; optional")
    parser.add_argument("--output-dir", default="outputs/train")
    parser.add_argument("--genes", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--radius", type=float, default=550)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--check-data", action="store_true",
                        help="Check barcode/gene alignment without loading UNI or training.")
    args = parser.parse_args(argv)
    for key in ("genes", "epochs"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    for key in ("lr", "radius"):
        value = getattr(args, key)
        if not np.isfinite(value) or value <= 0:
            parser.error(f"--{key} must be finite and positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    root = project_path(args.root)
    val_root = project_path(args.val_root) if args.val_root else None
    dataset, val_dataset = prepare_datasets(root, args.genes, args.radius, val_root)
    gene_names = dataset.gene_names
    print(f"Selected {len(gene_names)} genes: {gene_names}", flush=True)
    if args.check_data:
        print("PASS: patch shapes, coordinates and barcode/gene alignment.")
        return

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    weights_path = ROOT / "ckpts" / "pytorch_model.bin"
    if not weights_path.is_file():
        raise FileNotFoundError(f"Download UNI first: {weights_path}")
    from model.mymodel import MyModel
    model = MyModel(feature_dim=1024, genes=len(gene_names)).to(device)
    loader = make_dataloader(dataset, batch_size=1, shuffle=True, num_workers=0)
    val_loader = (make_dataloader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
                  if val_dataset is not None else None)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr,
    )
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = {**vars(args), "root": str(root), "val_root": str(val_root) if val_root else None,
                "weights_path": str(weights_path),
                "slides": [path.stem for path in dataset.path],
                "val_slides": [path.stem for path in val_dataset.path] if val_dataset is not None else [],
                "gene_names": gene_names, "label_transform": "raw", "adjacency": "unnormalized",
                "epoch_loss": "mean of per-slide MSE", "gene_selection": "training slides only",
                "best_checkpoint_metric": "val_mse" if val_loader is not None else None}
    (output_dir / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print(f"device={device}, train_slides={len(dataset)}, "
          f"val_slides={len(val_dataset) if val_dataset is not None else 0}, genes={len(gene_names)}", flush=True)
    loss_history = []
    val_loss_history = []
    for epoch in range(args.epochs):
        epoch_loss = train_one_epoch(model, loader, optimizer, device, epoch, args.epochs)
        loss_history.append(epoch_loss)
        message = f"Epoch [{epoch + 1}/{args.epochs}] Train MSE: {epoch_loss:.6f}"
        if val_loader is not None:
            val_loss = validate_one_epoch(model, val_loader, device, epoch, args.epochs)
            val_loss_history.append(val_loss)
            message += f" | Val MSE: {val_loss:.6f}"
        print(message, flush=True)
        save_epoch(output_dir, model, optimizer, epoch + 1, gene_names, settings,
                   loss_history, val_loss_history)


if __name__ == "__main__":
    main()
