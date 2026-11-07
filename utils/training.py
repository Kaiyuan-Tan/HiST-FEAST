"""Shared raw-count loss and gradient checks for both training entry points."""
import torch
from torch.nn import functional as F


def validate_labels(labels):
    if not torch.isfinite(labels).all() or (labels < 0).any():
        raise ValueError("Labels must be finite, nonnegative stored counts.")


def checked_mse_loss(predictions, labels, *, phase):
    """Keep shape mismatches from silently broadcasting in MSE."""
    if predictions.shape != labels.shape:
        raise ValueError(f"Prediction shape {predictions.shape} != label shape {labels.shape}")
    loss = F.mse_loss(predictions, labels)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"{phase} loss is NaN or Inf.")
    return loss


def check_gradients(model):
    if any(p.grad is not None and not torch.isfinite(p.grad).all()
           for p in model.parameters() if p.requires_grad):
        raise FloatingPointError("Training gradients contain NaN or Inf.")
