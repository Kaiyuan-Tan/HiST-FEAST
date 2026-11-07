"""Plot HEST GT and model predictions side by side on one shared color scale.

python visualization/compare_vis.py --slide NCBI338 --gene Map2
python visualization/compare_vis.py --slide NCBI338 --gene Map2 Sox17
"""
import argparse
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import project_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="hest_data", help="Dataset containing patches/ and st/.")
    parser.add_argument("--checkpoint", default="server_results/outputs/train_patch_mouse/best.pt")
    parser.add_argument("--uni-weights", default="ckpts/pytorch_model.bin")
    parser.add_argument("--slide", default="NCBI338")
    parser.add_argument("--gene", nargs="+", default=["Map2"], help="One or more case-sensitive target genes.")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--uni-batch-size", type=int, default=16)
    parser.add_argument("--vmin", type=float, help="Fixed shared lower limit; default includes zero.")
    parser.add_argument("--vmax", type=float, help="Fixed shared upper limit; default is max(GT, prediction).")
    args = parser.parse_args(argv)
    if args.uni_batch_size <= 0:
        parser.error("--uni-batch-size must be positive")
    for name in ("vmin", "vmax"):
        value = getattr(args, name)
        if value is not None and not math.isfinite(value):
            parser.error(f"--{name} must be finite")
    lower = args.vmin if args.vmin is not None else 0
    if args.vmax is not None and args.vmax <= lower:
        parser.error("--vmax must be greater than --vmin")
    return args


def read_coordinates(patch_path, expression_path):
    """Use expression-space coordinates in exactly the patch barcode order."""
    import anndata as ad
    import h5py
    import numpy as np

    with h5py.File(patch_path, "r") as handle:
        key = "barcodes" if "barcodes" in handle else "barcode"
        barcodes = handle[key].asstr()[:].reshape(-1)
        if len(barcodes) != len(handle["img"]):
            raise ValueError("Patch/barcode length mismatch.")
    if len(np.unique(barcodes)) != len(barcodes):
        raise ValueError("Duplicate patch barcodes must be resolved first.")
    data = ad.read_h5ad(expression_path, backed="r")
    try:
        if not data.obs_names.is_unique or not data.var_names.is_unique:
            raise ValueError("Duplicate expression barcodes or gene names.")
        if (data.obs_names.get_indexer(barcodes) < 0).any():
            raise ValueError("Some patch barcodes have no matching expression row.")
        if "spatial" not in data.obsm:
            raise ValueError("Expression data has no obsm['spatial'] coordinates.")
        return np.asarray(data[barcodes].obsm["spatial"], dtype=float)
    finally:
        data.file.close()


def main(argv=None):
    args = parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import torch
    from torchvision.transforms import PILToTensor

    from dataloader.HESTDataset import HESTDataset
    from utils.checkpoints import load_checkpoint
    from utils.features import extract_uni_features
    from utils.load_model import build_model, resolve_device
    from utils.plot_expression import plot_expression_comparison, save_expression_plot
    from utils.training import validate_labels

    checkpoint = load_checkpoint(project_path(args.checkpoint))
    gene_names = checkpoint["gene_names"]
    requested = list(dict.fromkeys(args.gene))
    missing = [gene for gene in requested if gene not in gene_names]
    if missing:
        raise ValueError(f"Model does not predict genes {missing}. Available genes: {gene_names}")
    radius = checkpoint["settings"]["radius"]
    dataset = HESTDataset(project_path(args.root), gene_names, PILToTensor(), radius=radius)
    index = next((i for i, path in enumerate(dataset.path) if path.stem == args.slide), None)
    if index is None:
        raise ValueError(f"Slide {args.slide!r} not found. Available slides: {[p.stem for p in dataset.path]}")
    coords = read_coordinates(dataset.path[index], dataset.gt_path[index])
    (images, adjacency), labels = dataset[index]
    validate_labels(labels)
    device = resolve_device(args.device)
    print(f"Predicting {args.slide}: patches={len(images)}, device={device}, radius={radius}", flush=True)
    model = build_model(checkpoint, uni_weights=project_path(args.uni_weights), device=device)
    del checkpoint
    with torch.no_grad():
        features = extract_uni_features(model.uni, images, device, args.uni_batch_size)
        prediction = model.resnet(model.gcn(features, adjacency.coalesce().to(device)))
    if prediction.shape != labels.shape:
        raise ValueError(f"Prediction shape {prediction.shape} != GT shape {labels.shape}")
    if not torch.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError("Predictions must be finite, nonnegative raw counts.")
    targets = labels.numpy()
    predictions = prediction.cpu().numpy()
    output_dir = project_path(args.output_dir)
    for gene in requested:
        column = gene_names.index(gene)
        fig, axes = plot_expression_comparison(
            coords, targets[:, column], predictions[:, column],
            title=f"{args.slide} · {gene}", vmin=args.vmin, vmax=args.vmax,
        )
        norm = axes[0].collections[0].norm
        output_path = save_expression_plot(fig, output_dir, args.slide, gene, "comparison")
        print(f"Saved: {output_path} (shared color limits: {norm.vmin:g}, {norm.vmax:g})", flush=True)


if __name__ == "__main__":
    main()
