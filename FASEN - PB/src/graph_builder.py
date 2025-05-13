import torch


# def build_knn_graph(X, k=10, symmetric=True):
#     """
#     构建 KNN 图的稠密邻接矩阵和热核参数。
#
#     Args:
#         X: 输入数据 (N, d)
#         k: 最近邻数量
#         symmetric: 是否对称化邻接矩阵
#
#     Returns:
#         knn_matrix: 稠密 KNN 邻接矩阵
#         chosen_t: 热核参数
#     """
#     k=k+1#由于对角线自身占一个位置，所以k=k+1
#     # 1. 计算欧氏距离矩阵
#     dist_matrix = torch.cdist(X, X, p=2)  # (N, N)
#
#     # 2. 计算热核参数 t
#     nonzero_distances = dist_matrix[dist_matrix > 0]
#     chosen_t = float(torch.median(nonzero_distances).item()) if len(nonzero_distances) > 0 else 1.0
#
#     # 3. 热核权重矩阵
#     t_sq = chosen_t * chosen_t
#     heat_kernel_matrix = torch.exp(-dist_matrix ** 2 / (2.0 * t_sq))  # (N, N)
#
#     # 4. KNN 邻接矩阵
#     knn_matrix = torch.zeros_like(heat_kernel_matrix)  # 初始化
#     top_k_values, top_k_indices = torch.topk(heat_kernel_matrix, k, dim=1, largest=True, sorted=False)
#     knn_matrix.scatter_(1, top_k_indices, top_k_values)  # 填入 KNN 的值
#
#     # 5. 对称化
#     if symmetric:
#         knn_matrix = (knn_matrix + knn_matrix.T) / 2.0
#
#     return knn_matrix, chosen_t

import torch
import numpy as np
import scipy.sparse as sp


def build_knn_graph(X, k=10, symmetric=True):
    """
    构建 KNN 图的稀疏邻接矩阵和热核参数。
    """
    k = k + 1  # 由于对角线自身占一个位置，所以 k = k + 1
    N = X.size(0)  # 样本数

    # 1. 计算欧氏距离矩阵
    dist_matrix = torch.cdist(X, X, p=2)  # (N, N)
    dist_matrix = dist_matrix.cpu().numpy()  # 转换为 NumPy 数组

    # 2. 排除对角线元素并计算热核参数 t
    np.fill_diagonal(dist_matrix, np.inf)  # 将对角线元素设为无穷大，防止影响计算
    # nonzero_distances = dist_matrix[dist_matrix < np.inf]  # 排除对角线的距离
    chosen_t = np.median(dist_matrix)

    # 3. 热核权重矩阵
    t_sq = chosen_t * chosen_t
    heat_kernel_matrix = np.exp(-dist_matrix ** 2 / (2.0 * t_sq))  # (N, N)

    # 4. KNN 邻接矩阵 (使用稀疏矩阵)
    row_indices = []
    col_indices = []
    values = []

    for i in range(N):
        top_k_values, top_k_indices = torch.topk(torch.tensor(heat_kernel_matrix[i]), k, dim=0, largest=True,
                                                 sorted=False)
        for idx, value in zip(top_k_indices, top_k_values):
            row_indices.append(i)
            col_indices.append(idx.item())
            values.append(value.item())

    knn_matrix = sp.coo_matrix((values, (row_indices, col_indices)), shape=(N, N))

    # 5. 对称化
    if symmetric:
        knn_matrix = (knn_matrix + knn_matrix.T) / 2.0

    return knn_matrix, chosen_t


import torch
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
import time

def build_euclidean_knn_graph(X, k=10, symmetric=True):
    """
    使用欧氏距离构建 KNN 图的稀疏邻接矩阵。

    Args:
        X (torch.Tensor): 输入样本张量，形状为 (N, D)，其中 N 为样本数量，D 为特征维度。
        k (int): 近邻数量。
        symmetric (bool): 是否对邻接矩阵进行对称化。

    Returns:
        knn_matrix (scipy.sparse.coo_matrix): 稀疏邻接矩阵 (N, N)。
        None: 仅返回邻接矩阵，与之前格式一致。
    """
    X_np = X  # 如果已经是 NumPy 数组

    # 使用 sklearn 计算 K 近邻
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric='euclidean').fit(X_np)  # k+1 是为了排除自己
    distances, indices = nbrs.kneighbors(X_np)  # indices 是 K 近邻索引

    # 构建稀疏矩阵的行索引、列索引和值
    row_indices = np.repeat(np.arange(X_np.shape[0]), k)
    col_indices = indices[:, 1:].reshape(-1)  # 排除对角线的自身索引
    values = distances[:, 1:].reshape(-1)  # 欧氏距离值

    # 构建稀疏邻接矩阵
    knn_matrix = sp.coo_matrix((values, (row_indices, col_indices)), shape=(X_np.shape[0], X_np.shape[0]))
    # print(f"Type of knn: {type(knn_matrix)}")
    # 对称化邻接矩阵

    #--------------------------------------------------------------------------------------------
    #第一种对称方案
    # start_time = time.time()  # 记录开始时间

    if symmetric:
        knn_matrix = (knn_matrix + knn_matrix.T) / 2.0#这里自动由coo转换为csr，为了优化计算

    # end_time = time.time()
    # print(f"第一种对称方案执行时间: {end_time - start_time:.4f} 秒")
    #--------------------------------------------------------------------------------------------
    # #第二种对称方案
    # start_time = time.time()
    #
    # if symmetric:
    #     # 使用原始的 row_indices 和 col_indices 进行拼接
    #     row_indices_new = np.concatenate([row_indices, col_indices])
    #     col_indices_new = np.concatenate([col_indices, row_indices])
    #     values_new = np.concatenate([values, values])
    #
    #     row_indices = row_indices_new
    #     col_indices = col_indices_new
    #     values = values_new
    #
    # end_time = time.time()
    # print(f"第一种对称方案执行时间: {end_time - start_time:.4f} 秒")


    return knn_matrix, None  # 保持返回格式一致

