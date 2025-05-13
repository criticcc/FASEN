import torch
import torch.nn as nn
import torch.nn.functional as F
import sympy
import scipy


def calculate_theta2(d):
    """
    Generate coefficients for wavelet transform
    """
    thetas = []
    x = sympy.symbols('x')
    for i in range(d + 1):
        f = sympy.poly((x / 2) ** i * (1 - x / 2) ** (d - i) / (scipy.special.beta(i + 1, d + 1 - i)))
        coeff = f.all_coeffs()
        inv_coeff = []
        for ii in range(d + 1):
            inv_coeff.append(float(coeff[d - ii]))
        thetas.append(inv_coeff)
    return thetas


class WaveletAutoEncoder(nn.Module):
    """
    Adaptive wavelet transform + multi-frequency input + encoder-decoder architecture
    """
    def __init__(self, input_dim, encoder_layers=3, encoder_hidden_dim=256,
                 fusion_dim=256, bottleneck_dim=128, wavelet_degree=2, freq_num=3):
        super(WaveletAutoEncoder, self).__init__()

        # Attention vectors for each layer, using nn.ParameterList
        self.att_vectors = nn.ModuleList([
            nn.ParameterList([nn.Parameter(torch.randn(encoder_hidden_dim)) for _ in range(freq_num)])
            for _ in range(encoder_layers)
        ])

        # Encoder modules
        self.encoders = nn.ModuleList([
            nn.ModuleList([nn.Linear(input_dim, encoder_hidden_dim) for _ in range(freq_num)])  # First layer
            if layer_idx == 0 else  # For other layers
            nn.ModuleList([nn.Linear(encoder_hidden_dim, encoder_hidden_dim) for _ in range(freq_num)])
            for layer_idx in range(encoder_layers)
        ])

        # Fusion layer
        self.fusion_fc = nn.Linear(freq_num * encoder_hidden_dim, fusion_dim)
        self.fusion_act = nn.LeakyReLU()

        # Bottleneck layer
        self.bottleneck_fc = nn.Linear(fusion_dim, bottleneck_dim)
        self.bottleneck_act = nn.LeakyReLU()

        # Decoder modules
        dec = []
        in_dim = bottleneck_dim
        for _ in range(encoder_layers):
            dec.append(nn.Linear(in_dim, encoder_hidden_dim))
            dec.append(nn.LeakyReLU())
            in_dim = encoder_hidden_dim
        dec.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*dec)

    def forward(self, freq_X_list):
        """
        Forward propagation, processing input data of different frequencies
        """

        batch_size = freq_X_list[0].size(0)

        previous_output = freq_X_list  # (freq_num, batch_size, input_dim)

        for l in range(len(self.att_vectors)):  # Iterate through each layer

            encoded_outputs = []  # Store encoded features for each frequency
            for i, x_i in enumerate(previous_output):  # Iterate through each frequency
                encoded_x_i = self.encoders[l][i](x_i)
                encoded_outputs.append(encoded_x_i)  # Save encoded features for each frequency

            # Weight the encoded features for each frequency
            weighted_outs = []

            # New addition: Apply softmax normalization to the attention vectors of the current layer
            current_att_vectors = list(self.att_vectors[l])
            att_matrix = torch.stack(current_att_vectors, dim=0)  # (freq_num, encoder_hidden_dim)
            normalized_att_matrix = F.softmax(att_matrix, dim=0)  # Softmax over the frequency dimension
            normalized_att_vectors = list(normalized_att_matrix.unbind(0))

            for i, encoded_x_i in enumerate(encoded_outputs):  # Iterate through each frequency's encoded output
                weighted_out = normalized_att_vectors[l][i] * encoded_x_i
                weighted_outs.append(weighted_out)

            previous_output = weighted_outs
            # Print the shape of previous_output
            # print(f"After Layer {l}, previous_output shape: {previous_output[0].shape}")
            # print(f"{l}th layer process over!")
        # Fusion
        cat_out = torch.cat(previous_output, dim=1)  # (batch_size, freq_num * encoder_hidden_dim)
        fused = self.fusion_act(self.fusion_fc(cat_out))

        # Bottleneck
        bottleneck = self.bottleneck_act(self.bottleneck_fc(fused))

        # Decoding
        recon = self.decoder(bottleneck)

        return recon