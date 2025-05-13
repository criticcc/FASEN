import torch
import torch.nn.functional as F
from torch.nn import MSELoss
from src.utils import calculate_auc_aupr_f1
from torch import optim
from src.optim_utils import Lamb, Lookahead, LookaheadAdam
from src.optim import LRScheduler
import numpy as np

def train(model, optimizer, train_data, train_origine, c):
    """
    Train the model, using reconstruction error from original training features as loss, and dynamically adjust learning rate with LRScheduler.

    Args:
        model: WaveletAutoEncoder model
        optimizer: Optimizer
        train_data: Training data (tensors with multi-frequency decomposition)
        train_origine: Original training data (tensors without frequency decomposition)
        c: Configuration parameters for initializing LRScheduler
    """
    # Initialize learning rate scheduler
    scheduler = LRScheduler(c=c, name=c.exp_scheduler, optimizer=optimizer)

    # Calculate steps per epoch
    num_steps_per_epoch = 1
    max_epochs = int(np.ceil(c.exp_num_total_steps / num_steps_per_epoch))

    print(f"Steps per epoch: {num_steps_per_epoch}")
    for epoch in range(max_epochs):
        model.train()  # Ensure model is in training mode
        for step in range(num_steps_per_epoch):
            optimizer.zero_grad()

            # Forward propagation
            train_output = model(train_data)

            # Calculate reconstruction error (MSE)
            mse = torch.norm(train_origine - train_output, dim=1)
            loss = mse.mean()
            print(f"Initial num_steps: {scheduler.num_steps}")

            # Backward propagation
            loss.backward()
            optimizer.step()  # Update model parameters
            scheduler.step()  # Update learning rate

        # Print training loss (once per epoch)
        print(f"Epoch [{epoch + 1}/{max_epochs}], Loss: {loss.item():.6f}")

    # Save model
    torch.save(model, "model.pth")
    print(f"Model saved to: model.pth")


# def train(model, optimizer, train_data, train_origine, epochs):
#     """
#     Train the model, using reconstruction error from original training features as loss, and save the trained model.
#
#     Args:
#         model: WaveletAutoEncoder model
#         optimizer: Optimizer
#         train_data: Training data (tensors with multi-frequency decomposition)
#         train_origine: Original training data (tensors without frequency decomposition)
#         epochs: Number of training epochs
#         save_path: Path to save the model, defaults to model.pth
#     """
#     for epoch in range(epochs):
#         model.train()
#         optimizer.zero_grad()
#
#         # Forward propagation
#         train_output = model(train_data)  # Model output
#
#         # Calculate reconstruction error (MSE)
#         mse = torch.norm(train_origine - train_output, dim=1)  # L2 norm per row
#         loss = mse.mean()  # Average as final loss
#
#         # Backward propagation
#         loss.backward()
#         optimizer.step()
#
#         # Print loss every 10 epochs
#         if (epoch + 1) % 10 == 0 or epoch == 0:
#             print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}")
#
#     # Save model
#     torch.save(model, "model.pth")
#     print(f"Model saved to: model.pth")

def evaluate(test_data, test_origine, test_label):
    """
    Evaluate model performance by loading the saved model.
    """
    # Load saved model
    model = torch.load('model.pth')
    model.eval()

    # Forward propagation
    test_output = model(test_data)

    # Calculate mean squared error
    mse = torch.norm(test_origine - test_output, dim=1).detach().cpu().numpy()

    # Calculate AUC, AUPR, and F1
    auc, aupr, f1 = calculate_auc_aupr_f1(test_label, mse)
    print(f"AUC: {auc:.4f}, AUPR: {aupr:.4f}, F1: {f1:.4f}")

    return auc, aupr, f1

def init_optimizer(c, model_parameters, device):
    # Initialize base optimizer
    if 'default' in c.exp_optimizer:
        print("Using Adam optimizer")  # Debug info
        optimizer = optim.Adam(params=model_parameters, lr=c.exp_lr)
    elif 'lamb' in c.exp_optimizer:
        print("Using Lamb optimizer")  # Debug info
        lamb = Lamb
        optimizer = lamb(
            model_parameters, lr=c.exp_lr, betas=(0.9, 0.999),
            weight_decay=c.exp_weight_decay, eps=1e-6)
    else:
        raise NotImplementedError(f"Optimizer {c.exp_optimizer} not implemented")

    # If Lookahead wrapper is enabled
    if c.exp_optimizer.startswith('lookahead_'):
        print("Using Lookahead optimizer")  # Debug info
        optimizer = Lookahead(optimizer, k=c.exp_lookahead_update_cadence)

    # Print debug info
    print(f"Optimizer initialized: {type(optimizer)}")
    return optimizer