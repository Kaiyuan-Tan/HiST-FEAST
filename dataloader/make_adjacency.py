import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors


def make_adjacency(coords, r):
    """
    coords: [N,2]，与 patch 顺序一致
    r: 距离阈值，单位与 coords 相同

    返回：包含权重为 1 的自连接、未归一化的邻接矩阵 [N,N]
    """
    coords = np.asarray(coords, dtype=np.float64)

    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) == 0:
        raise ValueError("coords must be [N, 2]")

    if not np.isfinite(coords).all():
        raise ValueError("coords must be limited")

    if not np.isfinite(r) or r <= 0:
        raise ValueError("r must be positive")

    neighbors = NearestNeighbors(radius=r).fit(coords)

    A = neighbors.radius_neighbors_graph(mode="connectivity")

    A = A.maximum(A.T).astype(np.uint8)
    A.setdiag(0)
    A.eliminate_zeros()
    A = A + sp.eye(A.shape[0], dtype=A.dtype, format="csr")

    return A