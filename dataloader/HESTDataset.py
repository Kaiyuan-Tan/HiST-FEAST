from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import scipy.sparse as sp
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from utils.make_adjacency import make_adjacency

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
        label = data[barcodes, self.gene_names].X
        if sp.issparse(label):
            label = label.toarray()
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
