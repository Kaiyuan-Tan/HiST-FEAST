from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import scipy.sparse as sp
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from dataloader.make_adjacency import make_adjacency


def read_gene_expression(data, barcodes, gene_names):
    """Align spots and sum raw expression columns with the same gene name."""
    if not data.obs_names.is_unique:
        raise ValueError("Expression data contains duplicate barcodes.")
    gene_names = list(gene_names)
    if not gene_names or len(set(gene_names)) != len(gene_names):
        raise ValueError("Target gene names must be nonempty and unique.")
    rows = data.obs_names.get_indexer(barcodes)
    if (rows < 0).any():
        raise ValueError("Patch barcodes are missing from expression data.")

    targets = {name: i for i, name in enumerate(gene_names)}
    target_columns = np.array([targets.get(name, -1) for name in data.var_names])
    columns = np.flatnonzero(target_columns >= 0)
    missing = set(gene_names).difference(data.var_names)
    if missing:
        raise ValueError(f"Missing target genes: {sorted(missing)}")

    # Integer positions work even when the AnnData column index is not unique.
    # Cast before summing so uint16 count matrices cannot overflow during grouping.
    # HDF5 dense datasets support only one fancy-indexed axis at a time.
    expression = (data[:, columns].X[rows, :] if data.isbacked
                  else data[rows, columns].X)
    expression = expression.astype(np.float32)
    grouping = sp.csr_matrix(
        (np.ones(len(columns), dtype=np.float32),
         (np.arange(len(columns)), target_columns[columns])),
        shape=(len(columns), len(gene_names)),
    )
    label = (grouping.T @ expression.T).T
    if sp.issparse(label):
        label = label.toarray()
    return np.asarray(label, dtype=np.float32)


class HESTDataset(Dataset):
    def __init__(self, root, gene_names, transform, radius=550):
        root = Path(root)
        self.root = root / "patches"
        self.gt = root / "st"
        self.transform = transform
        self.r = radius
        self.gene_names = list(gene_names)
        self.num_genes = len(self.gene_names)
        self.path = sorted(self.root.glob("*.h5"))
        self.gt_path = [self.gt / f"{path.stem}.h5ad" for path in self.path]

    def __len__(self):
        return len(self.path)

    def __getitem__(self, index):
        with h5py.File(self.path[index], "r") as f:
            key = "barcodes" if "barcodes" in f else "barcode"
            barcodes = f[key].asstr()[:].reshape(-1)
            coords = f["coords"][:]

            images = []
            for patch in f["img"]:
                patch = Image.fromarray(patch).convert("RGB")
                images.append(self.transform(patch))
            image = torch.stack(images)  # [N,3,H,W]

        # 按 patch 的 barcode 顺序和指定的基因顺序读取标签
        data = ad.read_h5ad(self.gt_path[index])
        label = read_gene_expression(data, barcodes, self.gene_names)
        label = torch.as_tensor(label, dtype=torch.float32)

        A = make_adjacency(coords, r=self.r).astype(np.float32).tocoo()
        # inv_degree = np.asarray(A.sum(axis=1)).ravel() ** -0.5
        # D_inv = sp.diags(inv_degree)
        # A = (D_inv @ A @ D_inv).tocoo()

        indices = torch.from_numpy(np.stack([A.row, A.col]).astype(np.int64))
        values = torch.from_numpy(A.data)
        A = torch.sparse_coo_tensor(indices, values, A.shape).coalesce()

        return ((image, A), label)


def collate_one_slide(batch):
    if len(batch) != 1:
        raise ValueError("一张切片一个样本，请设置 batch_size=1")
    return batch[0]


def make_dataloader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=None,
):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_one_slide if collate_fn is None else collate_fn,
    )
