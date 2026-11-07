import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slide", default="NCBI338", help="Slide ID (default: NCBI338).")
    parser.add_argument("--gene", default="Map2", help="Case-sensitive model target gene (default: Map2).")
    args = parser.parse_args(argv)

    # Support direct execution: python visualization/prediction_vis.py.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import anndata as ad
    import h5py
    import matplotlib
    import numpy as np
    import torch
    from torchvision.transforms import PILToTensor

    matplotlib.use("Agg")

    from dataloader.HESTDataset import HESTDataset
    from utils.features import extract_uni_features
    from utils.load_model import load_model
    from utils.plot_expression import plot_expression, save_expression_plot

    model, gene_names = load_model(
        checkpoint_path=ROOT / "server_results/outputs/train_patch_mouse/best.pt",
        uni_weights=ROOT / "ckpts/pytorch_model.bin",
    )
    if args.gene not in gene_names:
        raise ValueError(f"Model does not predict gene {args.gene!r}. Available genes: {gene_names}")
    gene_index = gene_names.index(args.gene)

    dataset = HESTDataset(
        root=ROOT / "hest_data",
        gene_names=gene_names,
        transform=PILToTensor(),
        radius=550,
    )
    index = next((i for i, p in enumerate(dataset.path) if p.stem == args.slide), None)
    if index is None:
        available = [p.stem for p in dataset.path]
        raise ValueError(f"Slide {args.slide!r} not found. Available slides: {available}")
    (images, adjacency), _ = dataset[index]
    device = next(model.parameters()).device

    with torch.no_grad():
        features = extract_uni_features(model.uni, images, device, batch_size=16)
        prediction = model.resnet(model.gcn(features, adjacency.to(device)))
    pred_values = prediction[:, gene_index].detach().cpu().numpy()

    with h5py.File(dataset.path[index], "r") as f:
        key = "barcode" if "barcode" in f else "barcodes"
        barcodes = f[key].asstr()[:].reshape(-1)
    adata = ad.read_h5ad(dataset.gt_path[index], backed="r")
    try:
        coords = np.asarray(adata[barcodes].obsm["spatial"])
    finally:
        adata.file.close()

    fig, ax = plot_expression(
        coords, pred_values, title=f"{args.slide} · {args.gene} prediction",
    )
    output_path = save_expression_plot(fig, ROOT / "output", args.slide, args.gene, "prediction")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
