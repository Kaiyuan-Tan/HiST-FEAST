"""Train with batched frozen UNI extraction and one complete graph per slide.

python train_patch.py --genes 100 --epochs 3 --uni-batch-size 64
python train_patch.py --check-data
python train_patch.py --root hest_data_splits/mouse/train --val-root hest_data_splits/mouse/val
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from dataloader.HESTDataset import make_dataloader
from train import ROOT, prepare_datasets, project_path, save_epoch


@torch.no_grad()
def extract_uni_features(uni, patches, device, batch_size):
    """Keep raw slide images on CPU; transfer and normalize one chunk at a time.

    Features retain the patch order used by adjacency and labels. Use no_grad,
    rather than inference_mode, so the trainable GCN can save them for backward.
    """
    if batch_size <= 0:
        raise ValueError("UNI batch size must be positive.")
    if patches.dtype != torch.uint8:
        raise TypeError("Use PILToTensor(): UNI expects raw uint8 images before /255.")
    if patches.ndim != 4 or len(patches) == 0:
        raise ValueError("Expected a nonempty batch of slide patches [N, C, H, W].")
    if any(p.requires_grad for p in uni.parameters()):
        raise ValueError("Batched feature extraction requires a frozen UNI.")
    uni.eval()
    features = None
    for start in range(0, len(patches), batch_size):
        batch = patches[start:start + batch_size].to(device)
        batch_features = uni(batch)
        if batch_features.ndim != 2 or batch_features.shape[0] != len(batch):
            raise ValueError("UNI must return one feature vector per patch [B, D].")
        if features is None:
            features = batch_features.new_empty((len(patches), batch_features.shape[1]))
        features[start:start + len(batch)].copy_(batch_features)
        del batch, batch_features
    return features


def train_one_epoch(model, loader, optimizer, device, epoch, epochs, uni_batch_size):
    model.train()
    model.uni.eval()
    total_loss = 0.0
    slides = 0
    progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}")

    for (patches, adjacency), labels in progress:
        if not torch.isfinite(labels).all() or (labels < 0).any():
            raise ValueError("Labels must be finite, nonnegative stored counts.")
        optimizer.zero_grad(set_to_none=True)

        # The loader's full uint8 slide stays on CPU, including during normalization.
        features = extract_uni_features(model.uni, patches, device, uni_batch_size)
        adjacency = adjacency.coalesce().to(device)
        labels = labels.to(device, dtype=torch.float32)
        # Aggregate all nodes together: splitting the graph would lose cross-chunk edges.
        predictions = model.resnet(model.gcn(features, adjacency))
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
        progress.set_postfix(mse=f"{loss.item():.6f}", patches=len(patches))
        del patches, features, adjacency, labels, predictions, loss

    if slides == 0:
        raise ValueError("No slides in the training loader.")
    return total_loss / slides


@torch.no_grad()
def validate_one_epoch(model, loader, device, epoch, epochs, uni_batch_size):
    model.eval()
    total_loss = 0.0
    slides = 0
    progress = tqdm(loader, desc=f"Val {epoch + 1}/{epochs}")
    for (patches, adjacency), labels in progress:
        if not torch.isfinite(labels).all() or (labels < 0).any():
            raise ValueError("Labels must be finite, nonnegative stored counts.")
        features = extract_uni_features(model.uni, patches, device, uni_batch_size)
        adjacency = adjacency.coalesce().to(device)
        labels = labels.to(device, dtype=torch.float32)
        predictions = model.resnet(model.gcn(features, adjacency))
        if predictions.shape != labels.shape:
            raise ValueError(f"Prediction shape {predictions.shape} != label shape {labels.shape}")
        loss = F.mse_loss(predictions, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Validation loss is NaN or Inf.")
        total_loss += loss.item()
        slides += 1
        progress.set_postfix(mse=f"{loss.item():.6f}", patches=len(patches))
        del patches, features, adjacency, labels, predictions, loss
    if slides == 0:
        raise ValueError("No slides in the validation loader.")
    return total_loss / slides


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="hest_data")
    parser.add_argument("--val-root", help="Validation directory containing patches/ and st/; optional")
    parser.add_argument("--output-dir", default="outputs/train_patch")
    parser.add_argument("--genes", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--radius", type=float, default=550)
    parser.add_argument("--uni-batch-size", type=int, default=64,
                        help="Number of image patches sent to frozen UNI per forward pass.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--check-data", action="store_true",
                        help="Check barcode/gene alignment without loading UNI or training.")
    args = parser.parse_args(argv)
    for key in ("genes", "epochs", "uni_batch_size"):
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
                "epoch_loss": "mean of per-slide MSE", "uni_extraction": "batched",
                "gene_selection": "training slides only",
                "best_checkpoint_metric": "val_mse" if val_loader is not None else None}
    (output_dir / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print(f"device={device}, train_slides={len(dataset)}, "
          f"val_slides={len(val_dataset) if val_dataset is not None else 0}, genes={len(gene_names)}, "
          f"uni_batch_size={args.uni_batch_size}", flush=True)
    loss_history = []
    val_loss_history = []
    for epoch in range(args.epochs):
        epoch_loss = train_one_epoch(
            model, loader, optimizer, device, epoch, args.epochs, args.uni_batch_size,
        )
        loss_history.append(epoch_loss)
        message = f"Epoch [{epoch + 1}/{args.epochs}] Train MSE: {epoch_loss:.6f}"
        if val_loader is not None:
            val_loss = validate_one_epoch(model, val_loader, device, epoch, args.epochs, args.uni_batch_size)
            val_loss_history.append(val_loss)
            message += f" | Val MSE: {val_loss:.6f}"
        print(message, flush=True)
        save_epoch(output_dir, model, optimizer, epoch + 1, gene_names, settings,
                   loss_history, val_loss_history)


if __name__ == "__main__":
    main()
