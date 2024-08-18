"""
Reduce overfitting by stopping training when diverging train and test scores.
->
Use PyTorch Lightning to implement a strategy called *early stopping*.

Early stopping can be implemented in PyTorch as a callback function.
That is, a callback function can be applied at certain stages
of the training process, such as at the end of each epoch.
PyTorch does not have any EarlyStopping class,
hence PyTorch Lightning's EarlyStopping is used.
"""
import lightning as pl  # PyTorch Lightning
import torch
import torch.nn as nn
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

MAX_EPOCHS = 100

# Create data with 10 features and 1000 observations
features, target = make_classification(
    n_classes=2,
    n_features=10,
    n_samples=1000,
    random_state=1,
)
# Because we are using simulated data using Scikit-Learn make_classification,
# we don't have to standardize the features.
# But in for real data, we must do standardization.

# Split training and test sets
features_train, features_test, target_train, target_test = train_test_split(
    features, target,
    test_size=0.1,
    random_state=1,
)
features_train.shape
# (900, 10)
features_test.shape
# (100, 10)

# Set random seed for PyTorch
torch.manual_seed(0)

# Convert data to PyTorch tensors
x_train = torch.from_numpy(features_train).float()
y_train = torch.from_numpy(target_train).float().view(-1, 1)
x_test = torch.from_numpy(features_test).float()
y_test = torch.from_numpy(target_test).float().view(-1, 1)


# Define a neural network using "Sequential"
class SequentialNN(nn.Module):
    """
    Feedforward two-layer neural network for binary classification using nn.Sequential.
    Each layer is "dense" (also called "fully connected")
    = All the units in the previous layer and in the next layer are connected.
    """
    def __init__(self):
        """ Initiates a network architecture. """
        super().__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(10, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.sequential(x)
        return x


class LightningNetwork(pl.LightningModule):
    def __init__(self, network):
        super().__init__()
        self.network = network
        # self.network = torch.compile(network)
        self.criterion = nn.BCELoss()
        self.metric = nn.functional.binary_cross_entropy

    def training_step(self, batch):
        """ Defines the train loop. """
        data, target = batch
        output = self.network(data)
        loss = self.criterion(output, target)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)


# Wrap data in TensorDataset
train_data = TensorDataset(x_train, y_train)
# Create data loader
train_loader = DataLoader(
    train_data,
    # batch_size = number of observations to propagate
    # through the network before updating the parameters
    batch_size=100,
    shuffle=True,  # schlurfen, mischen
)

# Initialize neural network
network = LightningNetwork(SequentialNN())

# Create trainer to monitor the test (validation) loss at each epoch,
# and if the test loss has not improved after three epochs (the default),
# training is interrupted.
trainer = pl.Trainer(
    callbacks=[
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=3,
        )
    ],
    max_epochs=MAX_EPOCHS,
)
# Train network
trainer.fit(model=network, train_dataloaders=train_loader)
"""
GPU available: False, used: False
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
...
  | Name      | Type         | Params | Mode 
---------------------------------------------------
0 | network   | SequentialNN | 465    | train
1 | criterion | BCELoss      | 0      | train
---------------------------------------------------
465       Trainable params
0         Non-trainable params
465       Total params
0.002     Total estimated model params size (MB)
9         Modules in train mode
0         Modules in eval mode
...
Epoch 15: 100%|██████████| 9/9 [00:00<00:00, 329.90it/s, v_num=9]
"""