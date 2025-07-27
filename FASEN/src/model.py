import torch
import torch.nn as nn
import torch.nn.functional as F
import sympy
import scipy


def calculate_theta2(d):
    """
    Generate wavelet transformation coefficients using Beta polynomials.
    """
    thetas = []
    x = sympy.symbols('x')
    for i in range(d + 1):
        f = sympy.poly((x / 2) ** i * (1 - x / 2) ** (d - i) / (scipy.special.beta(i + 1, d + 1 - i)))
        coeff = f.all_coeffs()
        inv_coeff = [float(coeff[d - ii]) for ii in range(d + 1)]
        thetas.append(inv_coeff)
    return thetas


class WaveletAutoEncoder(nn.Module):
    def __init__(self, input_dim, encoder_layers=3, encoder_hidden_dim=256,
                 fusion_dim=256, bottleneck_dim=128, wavelet_degree=2, freq_num=3):
        super(WaveletAutoEncoder, self).__init__()

        self.encoder_layers = encoder_layers
        self.freq_num = freq_num
        self.input_dim = input_dim
        self.encoder_hidden_dim = encoder_hidden_dim

        # Global learnable attention weights over frequency branches
        self.freq_weights = nn.Parameter(torch.ones(freq_num))

        # Encoder for each frequency component (independent MLPs)
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, encoder_hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_dim, encoder_hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_dim, encoder_hidden_dim)
            )
            for _ in range(freq_num)
        ])

        # Fusion layer after concatenating encoded outputs
        self.fusion_fc = nn.Linear(freq_num * encoder_hidden_dim, fusion_dim)
        self.fusion_act = nn.LeakyReLU()

        # Bottleneck (low-dimensional latent representation)
        self.bottleneck_fc = nn.Linear(fusion_dim, bottleneck_dim)
        self.bottleneck_act = nn.LeakyReLU()

        # Decoder (shared across all frequencies)
        dec = []
        in_dim = bottleneck_dim
        for _ in range(encoder_layers):
            dec.append(nn.Linear(in_dim, encoder_hidden_dim))
            dec.append(nn.LeakyReLU())
            in_dim = encoder_hidden_dim
        dec.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*dec)

    def forward(self, freq_X_list, flag=0):
        """
        Forward pass with frequency-aware attention and encoder-decoder reconstruction.
        Args:
            freq_X_list: list of tensors, each of shape (B, input_dim)
            flag: if set to 1, prints learnable frequency weights
        Returns:
            recon: reconstructed signal
        """
        batch_size = freq_X_list[0].size(0)

        # Softmax attention over frequency channels (global weights)
        temperature = 0.1
        att_weights = F.softmax(self.freq_weights / temperature, dim=0)

        # Apply attention to each frequency component
        weighted_input = [
            att_weights[i] * freq_X_list[i]
            for i in range(self.freq_num)
        ]

        # Encode each frequency component independently
        encoded_outputs = []
        for i, x_i in enumerate(weighted_input):
            encoded_x_i = self.encoders[i](x_i)
            encoded_outputs.append(encoded_x_i)

        # Concatenate and fuse
        cat_out = torch.cat(encoded_outputs, dim=1)
        fused = self.fusion_act(self.fusion_fc(cat_out))

        # Bottleneck
        bottleneck = self.bottleneck_act(self.bottleneck_fc(fused))

        # Decode
        recon = self.decoder(bottleneck)

        # Optionally print attention weights
        if flag == 1:
            with torch.no_grad():
                print("=== Learnable Frequency Weights ===")
                print(att_weights)

        return recon
