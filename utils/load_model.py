"""Rebuild the trainable model heads for evaluation and visualization."""
import torch


def resolve_device(device=None):
    """Accept an explicit device, or select CUDA/CPU for None and 'auto'."""
    if device is None or device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def build_model(checkpoint, uni_weights=None, device=None):
    """Build an evaluation model from an already loaded checkpoint."""
    # Keep metadata-only checks independent of timm and UNI construction.
    from model.mymodel import MyModel

    model = MyModel(feature_dim=1024, genes=len(checkpoint["gene_names"]),
                    weights_path=uni_weights)
    model.gcn.load_state_dict(checkpoint["gcn_state_dict"], strict=True)
    model.resnet.load_state_dict(checkpoint["resnet_state_dict"], strict=True)
    model.to(resolve_device(device))
    model.eval()
    return model


def load_model(checkpoint_path, uni_weights=None, device=None):
    """Load saved heads and return the model and its target gene order."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_model(checkpoint, uni_weights=uni_weights, device=device)
    return model, checkpoint["gene_names"]
