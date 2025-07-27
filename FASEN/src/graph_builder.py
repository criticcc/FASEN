import torch
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import pairwise_distances



def build_affinity_knn_graph(X, k=10, symmetric=True):
    """
    Build an affinity graph with heat kernel weights and KNN sparsification.

    Args:
        X (np.ndarray or torch.Tensor): Input data (N, D)
        k (int): Number of nearest neighbors to keep
        symmetric (bool): Whether to symmetrize the matrix

    Returns:
        affinity_knn (scipy.sparse.coo_matrix): Sparse affinity graph
        None: Placeholder
    """
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()

    dist_matrix = pairwise_distances(X, metric='euclidean')
    median_dist = np.median(dist_matrix[dist_matrix > 0])
    sigma = 1  # or sigma = median_dist

    affinity_matrix = np.exp(-dist_matrix ** 2 / (2 * sigma ** 2))

    N = affinity_matrix.shape[0]
    row_indices, col_indices, values = [], [], []

    for i in range(N):
        row = affinity_matrix[i].copy()
        row[i] = 0  # exclude self
        knn_indices = np.argpartition(-row, k)[:k]
        for j in knn_indices:
            row_indices.append(i)
            col_indices.append(j)
            values.append(row[j])

    affinity_knn = sp.coo_matrix((values, (row_indices, col_indices)), shape=(N, N))

    if symmetric:
        affinity_knn = (affinity_knn + affinity_knn.T) / 2.0

    return affinity_knn, None
