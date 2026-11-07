"""Batched feature extraction shared by training, evaluation and visualization."""
import torch


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
