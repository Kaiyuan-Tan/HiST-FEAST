"""Shared target selection and HEST dataset checks for training and evaluation."""
import anndata as ad
import h5py
import numpy as np
from torchvision.transforms import PILToTensor

from dataloader.HESTDataset import HESTDataset

CONTROL_PREFIXES = ("BLANK_", "NegControlCodeword_", "NegControlProbe_")


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
