import torch
import torch.nn.functional as F
from torch.nn import MSELoss
from src.utils import calculate_auc_aupr_f1
from torch import optim
from src.optim_utils import Lamb, Lookahead, LookaheadAdam
from src.optim import LRScheduler
import numpy as np
from collections import defaultdict, OrderedDict

class Trainer:
    def __init__(self, model, c, train_set, train_origine, test_set, test_origine, device):
        """
        Initialize the Trainer class.

        Args:
            model: WaveletAutoEncoder model.
            c: Configuration parameters.
            train_set: Training dataset.
            test_set: Testing dataset.
            train_origine: Original training data.
            test_origine: Original testing data.
            device: Device type (cuda or cpu).
        """
        self.model = model
        self.c = c
        self.train_set = train_set
        self.test_set = test_set
        self.optimizer = self.init_optimizer(c, model.parameters(), device=device)
        self.optimizer._optimizer_step_pre_hooks = OrderedDict()
        self.optimizer._optimizer_step_post_hooks = OrderedDict()  # Required for newer versions of torch to avoid errors
        print(f'Initialized "{c.exp_optimizer}" optimizer.')
        self.train_loader, self.test_loader = get_dataloader()

    def training(self):
        """
        Train the model.
        """
        # Initialize learning rate scheduler
        self.model.train()
        scheduler = LRScheduler(c=self.c, name=self.c.exp_scheduler, optimizer=self.optimizer)

        # Calculate steps per epoch
        num_steps_per_epoch = 1
        max_epochs = int(np.ceil(self.c.exp_num_total_steps / num_steps_per_epoch))

        print(f"Steps per epoch: {num_steps_per_epoch}")
        for epoch in range(max_epochs):
            for step in range(num_steps_per_epoch):
                self.optimizer.zero_grad()

                # Forward propagation
                train_data, train_origine = self.train_set
                train_output = self.model(train_data)

                # Calculate reconstruction error (MSE)
                mse = torch.norm(train_origine - train_output, dim=1)
                loss = mse.mean()
                print(f"Initial num_steps: {scheduler.num_steps}")

                # Backward propagation
                loss.backward()
                self.optimizer.step()  # Update model parameters
            scheduler.step()  # Update learning rate

            # Print training loss (once per epoch)
            print(f"Epoch [{epoch + 1}/{max_epochs}], Loss: {loss.item():.6f}")

        # Save model
        torch.save(self.model, "model.pth")
        print(f"Model saved to: model.pth")

    def evaluate(self):
        """
        Evaluate model performance.
        """
        model = torch.load('model.pth')
        model.eval()

        test_data, test_origine, test_label = self.test_set

        # Forward propagation
        test_output = model(test_data)

        # Calculate mean squared error
        mse = torch.norm(test_origine - test_output, dim=1).detach().cpu().numpy()

        # Calculate AUC, AUPR, and F1
        auc, aupr, f1 = calculate_auc_aupr_f1(test_label, mse)
        print(f"AUC: {auc:.4f}, AUPR: {aupr:.4f}, F1: {f1:.4f}")

        return auc, aupr, f1

    def init_optimizer(self, c, model_parameters, device):
        """
        Initialize the optimizer.

        Args:
            c: Configuration parameters.
            model_parameters: Model parameters.
            device: Device type.

        Returns:
            optimizer: Optimizer object.
        """
        if 'default' in c.exp_optimizer:
            print("Using Adam optimizer")
            optimizer = optim.Adam(params=model_parameters, lr=c.exp_lr)
        elif 'lamb' in c.exp_optimizer:
            print("Using Lamb optimizer")
            lamb = Lamb
            optimizer = lamb(
                model_parameters, lr=c.exp_lr, betas=(0.9, 0.999),
                weight_decay=c.exp_weight_decay, eps=1e-6)
        else:
            raise NotImplementedError(f"Optimizer {c.exp_optimizer} not implemented")

        if c.exp_optimizer.startswith('lookahead_'):
            print("Using Lookahead optimizer")
            optimizer = Lookahead(optimizer, k=c.exp_lookahead_update_cadence)

        print(f"Optimizer initialized: {type(optimizer)}")
        return optimizer