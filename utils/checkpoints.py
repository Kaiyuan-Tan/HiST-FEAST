"""Read and save training checkpoints without constructing UNI."""
import json
import math

import torch


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


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    required = {"gene_names", "settings", "gcn_state_dict", "resnet_state_dict"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError("Expected a checkpoint saved by train.py or train_patch.py.")
    genes = checkpoint["gene_names"]
    if (not isinstance(genes, list) or not genes
            or any(not isinstance(name, str) or not name for name in genes)
            or len(set(genes)) != len(genes)):
        raise ValueError("Checkpoint gene_names must be a nonempty list of unique names.")
    settings = checkpoint["settings"]
    if not isinstance(settings, dict):
        raise ValueError("Invalid checkpoint settings.")
    if settings.get("label_transform") != "raw" or settings.get("adjacency") != "unnormalized":
        raise ValueError("This evaluator requires raw-count labels and unnormalized adjacency.")
    radius = settings.get("radius")
    if not isinstance(radius, (int, float)) or not math.isfinite(radius) or radius <= 0:
        raise ValueError("Checkpoint must contain the training graph radius.")
    for field in ("slides", "val_slides"):
        names = settings.get(field, [])
        if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
            raise ValueError(f"Invalid {field} in checkpoint settings.")
    if not settings.get("slides"):
        raise ValueError("Checkpoint lacks training slide IDs.")
    return checkpoint
