import numpy as np
import torch
from sklearn.model_selection import train_test_split

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.graph_builder import build_knn_graph, build_euclidean_knn_graph
from src.model import calculate_theta2
import scipy.sparse as sp
import time


def preprocess_data2(data, wavelet_degree, k):
    X, y = data
    train_X, train_label, test_X, test_label = split_X(X, y)
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)
    train_knn_matrix, train_chosen_t = build_euclidean_knn_graph(train_X, k=k, symmetric=True)
    train_laplacian = calculate_laplacian(train_knn_matrix)
    train_freq_X_list = apply_wavelet_filter(train_X, train_laplacian, wavelet_degree)
    test_knn_matrix, test_chosen_t = build_euclidean_knn_graph(test_X, k=k, symmetric=True)
    test_laplacian = calculate_laplacian(test_knn_matrix)
    test_freq_X_list = apply_wavelet_filter(test_X, test_laplacian, wavelet_degree)
    return train_freq_X_list, test_freq_X_list, train_X, test_X, train_label, test_label


def preprocess_data_draw(data, wavelet_degree, k):
    X, y = data
    train_X, train_label, test_X, test_label = split_X(X, y)
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)
    train_knn_matrix, train_chosen_t = build_euclidean_knn_graph(train_X, k=k, symmetric=True)
    train_laplacian = calculate_laplacian(train_knn_matrix)
    test_knn_matrix, test_chosen_t = build_euclidean_knn_graph(test_X, k=k, symmetric=True)
    test_laplacian = calculate_laplacian(test_knn_matrix)
    return train_X, test_X, train_label, test_label, train_laplacian, test_laplacian


def split_X(X, labels):
    inlier_indices = np.where(labels == 0)[0]
    outlier_indices = np.where(labels == 1)[0]
    num_inliers = len(inlier_indices)
    num_split = num_inliers // 2
    train_X = X[inlier_indices[:num_split]]
    train_label = np.zeros(num_split)
    test_X = X[np.concatenate([inlier_indices[num_split:], outlier_indices])]
    test_label = np.zeros(len(test_X))
    test_label[len(inlier_indices[num_split:]):] = 1
    return train_X, train_label, test_X, test_label


from scipy.sparse import coo_matrix, diags
import torch
import numpy as np
import scipy.sparse as sp


def calculate_laplacian(knn_matrix):
    degree_matrix = sp.diags(knn_matrix.sum(axis=1).A1)
    degree_inv_sqrt = sp.diags(1.0 / (np.sqrt(degree_matrix.diagonal()) + 1e-6))
    laplacian = sp.eye(knn_matrix.shape[0]) - degree_inv_sqrt @ knn_matrix @ degree_inv_sqrt
    return laplacian


from scipy.sparse import csr_matrix, lil_matrix
import scipy.sparse.linalg as splin
from scipy.sparse import eye


def apply_wavelet_filter(X, laplacian, wavelet_degree, block_size=512):
    thetas = calculate_theta2(wavelet_degree)
    freq_X_list = []
    laplacian_powers = [eye(laplacian.shape[0], format='csr')]
    for i in range(1, wavelet_degree + 1):
        laplacian_powers.append(laplacian_powers[-1].dot(laplacian))
    for theta in thetas:
        filter_matrix = lil_matrix(laplacian.shape)
        for t, lp in zip(theta, laplacian_powers):
            num_rows = lp.shape[0]
            for start in range(0, num_rows, block_size):
                end = min(start + block_size, num_rows)
                lp_block = lp[start:end, :]
                filter_matrix[start:end, :] += t * lp_block
        filter_matrix = csr_matrix(filter_matrix)
        d = X.shape[1]
        result_blocks = []
        for start in range(0, d, block_size):
            end = min(start + block_size, d)
            result_block = filter_matrix.dot(X[:, start:end])
            result_blocks.append(result_block)
        X_filtered = np.hstack(result_blocks)
        X_filtered = torch.tensor(X_filtered).to('cuda')
        freq_X_list.append(X_filtered)
    X = torch.tensor(X)
    Xx = X.clone().detach().to('cuda')
    freq_X_list.append(Xx)
    return freq_X_list


def load_data(filepath):
    if filepath.endswith('.npz'):
        return load_npz_data(filepath)
    elif filepath.endswith('.csv'):
        return load_csv_data(filepath)
    elif filepath.endswith('.mat'):
        return load_mat_data(filepath)
    else:
        raise ValueError("Unsupported file format. Supported formats are: .npz, .csv, .mat")


def load_npz_data(filepath):
    data = np.load(filepath)
    X = data['X']
    y = data['y']
    return X, y


def load_csv_data(filepath):
    data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
    X = data[:, :-1]
    y = data[:, -1]
    return X, y


def load_mat_data(filepath):
    from scipy.io import loadmat
    data = loadmat(filepath)
    X = data['X']
    y = data['y'].reshape(-1)
    return X, y


import torch
import numpy as np


def split_freq_X_list(freq_X_list, X, labels):
    inlier_indices = np.where(labels == 0)[0]
    outlier_indices = np.where(labels == 1)[0]
    num_inliers = len(inlier_indices)
    num_split = num_inliers // 2
    train_data = [freq_X_list[i][inlier_indices[:num_split]] for i in range(len(freq_X_list))]
    train_label = np.zeros(num_split)
    test_data = [freq_X_list[i][np.concatenate([inlier_indices[num_split:], outlier_indices])] for i in
                 range(len(freq_X_list))]
    test_label = np.zeros(len(test_data[0]))
    test_label[len(inlier_indices[num_split:]):] = 1
    train_origine = X[inlier_indices[:num_split]]
    test_origine = X[np.concatenate([inlier_indices[num_split:], outlier_indices])]
    return train_data, train_label, test_data, test_label, train_origine, test_origine


import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve, precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler


def calculate_auc_aupr_f1(test_label, mse):
    scaler = MinMaxScaler()
    mse_normalized = scaler.fit_transform(mse.reshape(-1, 1)).flatten()
    def aucPerformance(score, labels):
        roc_auc = roc_auc_score(labels, score)
        ap = average_precision_score(labels, score)
        return roc_auc, ap
    auc, aupr = aucPerformance(mse_normalized, test_label)
    def F1Performance(score, target):
        normal_ratio = (target == 0).sum() / len(target)
        score = np.squeeze(score)
        threshold = np.percentile(score, 100 * normal_ratio)
        pred = np.zeros(len(score))
        pred[score > threshold] = 1
        _, _, f1, _ = precision_recall_fscore_support(target, pred, average='binary')
        return f1
    f1 = F1Performance(mse_normalized, test_label)
    return auc, aupr, f1