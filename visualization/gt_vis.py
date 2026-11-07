"""Plot a selected HEST slide's measured gene expression and save a PNG."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slide", default="NCBI338", help="Slide ID (default: NCBI338).")
    parser.add_argument("--gene", default="Map2", help="Case-sensitive gene name (default: Map2).")
    args = parser.parse_args(argv)

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import anndata as ad
    import h5py
    import matplotlib
    import numpy as np
    from scipy import sparse

    matplotlib.use("Agg")

    from utils.plot_expression import plot_expression, save_expression_plot

    data_dir = ROOT / "hest_data"
    patch_path = data_dir / "patches" / f"{args.slide}.h5"
    expression_path = data_dir / "st" / f"{args.slide}.h5ad"
    for path in (patch_path, expression_path):
        if not path.is_file():
            raise FileNotFoundError(f"Slide {args.slide!r}: missing {path}")

    with h5py.File(patch_path, "r") as f:
        key = "barcode" if "barcode" in f else "barcodes"
        barcodes = f[key].asstr()[:].reshape(-1)
        if len(barcodes) != len(f["img"]):
            raise ValueError("Patch/barcode length mismatch.")
    if len(np.unique(barcodes)) != len(barcodes):
        raise ValueError("Duplicate patch barcodes must be resolved first.")

    adata = ad.read_h5ad(expression_path)
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError("Duplicate expression barcodes or gene names.")
    if args.gene not in adata.var_names:
        raise ValueError(f"Gene {args.gene!r} not found in slide {args.slide!r}.")
    if (adata.obs_names.get_indexer(barcodes) < 0).any():
        raise ValueError("Some patch barcodes have no matching expression row.")
    if "spatial" not in adata.obsm:
        raise ValueError("Expression data has no obsm['spatial'] coordinates.")

    # Match the same patch order and plotting locations used by prediction_vis.py.
    aligned = adata[barcodes, [args.gene]]
    coords = np.asarray(aligned.obsm["spatial"], dtype=float)
    column = aligned.X
    true_values = np.asarray(
        column.toarray() if sparse.issparse(column) else column, dtype=float,
    ).reshape(-1)

    # Use the stored scale, matching prediction_vis.py's raw-count display.
    fig, ax = plot_expression(
        coords, true_values, title=f"{args.slide} · {args.gene} GT",
    )
    output_path = save_expression_plot(fig, ROOT / "output", args.slide, args.gene, "GT")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
