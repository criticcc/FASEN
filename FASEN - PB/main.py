import os
import torch
import torch.optim as optim
import pandas as pd
from src.model import WaveletAutoEncoder
from src.utils import load_data, split_freq_X_list, calculate_auc_aupr_f1, preprocess_data2
from src.trainer import train, evaluate
from src.trainer import init_optimizer
from src.config import build_parser
from collections import defaultdict, OrderedDict
import numpy as np

# Configuration parameters
batch_size = 512
epochs = 300
# learning_rate = 1e-5
weight_decay = 1e-5
encoder_layers = 2
encoder_hidden_dim = 256
fusion_dim = 256
bottleneck_dim = 32
random_seed = 42

parser = build_parser()
c = parser.parse_args()

np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = "data"
data_files = [f for f in os.listdir(data_dir) if f.endswith(('.npz', '.csv', '.mat'))]

# Save best results
best_results = []

# Grid search hyperparameters
k_values = list(range(5, 35, 5))
wavelet_degrees = list(range(2, 18, 2))
lr = [0.1, 0.01, 0.001, 0.0001]

for dataset_name in data_files:
    print(f"Processing data file: {dataset_name}...")

    X, y = load_data(os.path.join(data_dir, dataset_name))

    print(f"Type of X: {X.dtype}")

    intermediate_results = []

    for wavelet_degree in wavelet_degrees:
        for k in k_values:
            print(f"Value of k: {k}")  # Check the value of k
            train_data, test_data, train_origine, test_origine, train_label, test_label = preprocess_data2((X, y), wavelet_degree, k)

            train_origine = torch.tensor(train_origine)
            test_origine = torch.tensor(test_origine)
            # Split dataset
            print("Splitting training and testing sets...")

            # Move data to GPU
            train_data = torch.stack(train_data).to(device).float()
            test_data = torch.stack(test_data).to(device).float()
            test_origine = test_origine.to(device)
            train_origine = train_origine.to(device)
            for learn_rate in lr:
                c.exp_lr = learn_rate
                print(f"Searching hyperparameters k={k}, wavelet_degree={wavelet_degree}...")

                print("Initializing model...")
                model = WaveletAutoEncoder(
                    input_dim=X.shape[1],
                    encoder_hidden_dim=encoder_hidden_dim,
                    encoder_layers=encoder_layers,
                    fusion_dim=fusion_dim,
                    bottleneck_dim=bottleneck_dim,
                    freq_num=wavelet_degree + 2,
                    wavelet_degree=wavelet_degree,
                ).to(device)

                optimizer = init_optimizer(c=c, model_parameters=model.parameters(), device=device)
                #
                optimizer._optimizer_step_pre_hooks = OrderedDict()
                optimizer._optimizer_step_post_hooks = OrderedDict()  # Required for newer versions of torch to avoid errors
                print(f'Initialized "{c.exp_optimizer}" optimizer.')
                # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

                # Model training
                print(f"Starting model training: {dataset_name}...")
                train(model, optimizer, train_data, train_origine, c)

                # Model testing
                print(f"Starting model testing: {dataset_name}...")
                auc, aupr, f1 = evaluate(test_data, test_origine, test_label)

                intermediate_results.append({
                    'k': k,
                    'wavelet_degree': wavelet_degree,
                    'AUC': auc,
                    'AUPR': aupr,
                    "f1": f1,
                    "lr": learn_rate
                })

                results = pd.DataFrame(intermediate_results)
                os.makedirs("result", exist_ok=True)
                results.to_csv(f"result/{dataset_name}_results.csv", index=False)

                print(
                    f"Dataset: {dataset_name}, k={k}, wavelet_degree={wavelet_degree}, AUC: {auc:.4f}, AUPR: {aupr:.4f}")

    best_result = max(intermediate_results, key=lambda x: x['AUPR'])

    best_results.append({
        'Dataset': dataset_name,
        'Best_k': best_result['k'],
        'Best_wavelet_degree': best_result['wavelet_degree'],
        'AUC': best_result['AUC'],
        'AUPR': best_result['AUPR'],
        "f1": best_result['f1'],
        "lr": best_result['lr']
    })

best_results_df = pd.DataFrame(best_results)
best_results_df.to_csv("result/best_result.csv", index=False)

print("All datasets processed, best results saved to result/best_result.csv")