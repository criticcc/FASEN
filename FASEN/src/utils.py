import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from src.graph_builder import build_affinity_knn_graph
from src.model import calculate_theta2
from scipy.sparse import eye
import scipy.sparse as sp
import time


class MultiFreqDataset(torch.utils.data.Dataset):
    def __init__(self, freq_X_list, origine_X, label=None):
        self.freq_X_list = freq_X_list
        self.origine_X = origine_X
        self.label = label
        self.n = origine_X.shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        freq_sample = [freq_X[idx] for freq_X in self.freq_X_list]
        origine_sample = self.origine_X[idx]
        if self.label is not None:
            return freq_sample, origine_sample, self.label[idx]
        else:
            return freq_sample, origine_sample


def preprocess_data2(data, config):
    k = config['k']
    wavelet_degree = config['wavelet_degree']
    batchsize = config['batchsize']

    X, y = data
    train_X, train_label, test_X, test_label = split_X(X, y)

    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)

    train_knn_matrix, _ = build_affinity_knn_graph(train_X, k=k, symmetric=True)
    train_laplacian = calculate_laplacian(train_knn_matrix)
    train_freq_X_list = apply_wavelet_filter(train_X, train_laplacian, wavelet_degree)

    test_knn_matrix, _ = build_affinity_knn_graph(test_X, k=k, symmetric=True)
    test_laplacian = calculate_laplacian(test_knn_matrix)
    test_freq_X_list = apply_wavelet_filter(test_X, test_laplacian, wavelet_degree)

    train_X = torch.tensor(train_X)
    test_X = torch.tensor(test_X)

    train_dataset = MultiFreqDataset(train_freq_X_list, train_X, train_label)
    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)

    test_dataset = MultiFreqDataset(test_freq_X_list, test_X, test_label)
    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)

    return train_loader, test_loader


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


def calculate_laplacian(knn_matrix):
    degree_matrix = sp.diags(knn_matrix.sum(axis=1).A1)
    degree_inv_sqrt = sp.diags(1.0 / (np.sqrt(degree_matrix.diagonal()) + 1e-6))
    laplacian = sp.eye(knn_matrix.shape[0]) - degree_inv_sqrt @ knn_matrix @ degree_inv_sqrt
    return laplacian


def apply_wavelet_filter(X, laplacian, wavelet_degree):
    start_time = time.time()
    thetas = calculate_theta2(wavelet_degree)
    freq_X_list = []
    laplacian_powers = [eye(laplacian.shape[0], format='csr')]
    for i in range(1, wavelet_degree + 1):
        laplacian_powers.append(laplacian_powers[-1].dot(laplacian))
    for theta in thetas:
        filter_matrix = sum(t * lp for t, lp in zip(theta, laplacian_powers))
        filter_dense = torch.tensor(filter_matrix.toarray(), dtype=torch.float32).to('cuda')
        X_dense = torch.tensor(X, dtype=torch.float32).to('cuda')
        X_filtered = torch.matmul(filter_dense, X_dense)
        freq_X_list.append(X_filtered)
    end_time = time.time()
    print(f"apply_wavelet_filter done in {end_time - start_time:.4f} sec")
    return freq_X_list


def load_data(filepath):
    if filepath.endswith('.npz'):
        return load_npz_data(filepath)
    elif filepath.endswith('.csv'):
        return load_csv_data(filepath)
    elif filepath.endswith('.mat'):
        return load_mat_data(filepath)
    else:
        raise ValueError("Unsupported file format.")


def load_npz_data(filepath):
    data = np.load(filepath, allow_pickle=True)
    return data['X'], data['y']


def load_csv_data(filepath):
    data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
    return data[:, :-1], data[:, -1]


def load_mat_data(filepath):
    from scipy.io import loadmat
    data = loadmat(filepath)
    return data['X'], data['y'].reshape(-1)


def split_freq_X_list(freq_X_list, X, labels):
    inlier_indices = np.where(labels == 0)[0]
    outlier_indices = np.where(labels == 1)[0]
    num_inliers = len(inlier_indices)
    num_split = num_inliers // 2

    train_data = [freq[i][inlier_indices[:num_split]] for i, freq in enumerate(freq_X_list)]
    train_label = np.zeros(num_split)

    test_indices = np.concatenate([inlier_indices[num_split:], outlier_indices])
    test_data = [freq[i][test_indices] for i, freq in enumerate(freq_X_list)]
    test_label = np.zeros(len(test_data[0]))
    test_label[len(inlier_indices[num_split:]):] = 1

    train_origine = X[inlier_indices[:num_split]]
    test_origine = X[test_indices]

    return train_data, train_label, test_data, test_label, train_origine, test_origine


from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler


def calculate_auc_aupr_f1(test_label, mse):
    scaler = MinMaxScaler()
    mse_normalized = scaler.fit_transform(mse.reshape(-1, 1)).flatten()

    auc = roc_auc_score(test_label, mse_normalized)
    aupr = average_precision_score(test_label, mse_normalized)

    normal_ratio = (test_label == 0).sum() / len(test_label)
    threshold = np.percentile(mse_normalized, 100 * normal_ratio)
    pred = (mse_normalized > threshold).astype(int)
    _, _, f1, _ = precision_recall_fscore_support(test_label, pred, average='binary')

    return auc, aupr, f1
