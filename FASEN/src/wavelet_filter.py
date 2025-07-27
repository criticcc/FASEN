# src/wavelet_filter.py
import torch
from src.model import calculate_theta2

def _poly_filter(X_t, L_dense, coefs):
    """
    sum_{k=0..len(coefs)-1} coefs[k]* (L^k X)
    """
    out = coefs[0]*X_t
    L_power = X_t.clone()
    for i in range(1,len(coefs)):
        L_power = L_dense @ L_power
        out += coefs[i]*L_power
    return out

def generate_freq_features(X, L_dense, wavelet_degree, freq_num, device):
    """
    1) compute thetas= (d+1)x(d+1)
    2) freq_num 行 => generate freq_list
    """
    X_t = torch.tensor(X, dtype=torch.float, device=device)
    L_dense = L_dense.to(device)

    thetas = calculate_theta2(wavelet_degree)  # (d+1)x(d+1)
    freq_list = []
    for i in range(freq_num):
        if i < len(thetas):
            coefs = thetas[i]
        else:
            coefs = thetas[-1]
        Xi = _poly_filter(X_t, L_dense, coefs)
        freq_list.append(Xi)
    return freq_list  # list of Tensors => freq_num
