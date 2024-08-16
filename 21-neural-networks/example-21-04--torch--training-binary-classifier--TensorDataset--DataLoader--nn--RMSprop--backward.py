"""
Train a binary classifier neural network.
->
Construct a feedforward neural network with an output layer using
the sigmoid (logistic) activation function in PyTorch and train it.

Notice:
python-dev for using Python API for C should be installed;
if not, install it on Ubuntu
$ sudo apt-get install python3.x-dev
where 3.x is your Python version in your virtual environment.
"""
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import RMSprop
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create data with 10 features and 1000 observations
features, target = make_classification(
    n_classes=2,
    n_features=10,
    n_samples=1000,
    random_state=1,
)
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

# Set random seeds
torch.manual_seed(0)
np.random.seed(0)

# Convert data to PyTorch tensors
x_train = torch.from_numpy(features_train).float()
y_train = torch.from_numpy(target_train).float().view(-1, 1)
x_test = torch.from_numpy(features_test).float()
y_test = torch.from_numpy(target_test).float().view(-1, 1)


# Define a neural network using "Sequential"
class SequentialNN(nn.Module):
    """
    Feedforward two-layer neural network for binary classifications using nn.Sequential.
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


# Create neural network
network = SequentialNN()

# Define loss function, optimizer
criterion = nn.BCELoss()
optimizer = RMSprop(network.parameters())

# Create data loader
train_data = TensorDataset(x_train, y_train)
train_loader = DataLoader(
    train_data,
    # batch_size = number of observations to propagate
    # through the network before updating the parameters
    batch_size=100,
    shuffle=True,
)

# Compile the model using torch 2.0's optimizer
network = torch.compile(network)

# Train neural network
for epoch_idx in range(3):  # how many epochs to use when training the data
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)  # using the forward method
        loss = criterion(output, target)
        loss.backward()  # to update the gradients
        optimizer.step()
        print("Epoch:", epoch_idx + 1, ";", "\tLoss:", loss.item())
# Epoch: 1 ; 	Loss: 0.6873747110366821
# Epoch: 1 ; 	Loss: 0.6832718849182129
# Epoch: 1 ; 	Loss: 0.6220871210098267
# Epoch: 1 ; 	Loss: 0.4690907299518585
# Epoch: 1 ; 	Loss: 0.3916798532009125
# Epoch: 1 ; 	Loss: 0.4014907479286194
# Epoch: 1 ; 	Loss: 0.3759861886501312
# Epoch: 1 ; 	Loss: 0.6534170508384705
# Epoch: 1 ; 	Loss: 0.42521628737449646
# Epoch: 2 ; 	Loss: 0.3898506164550781
# Epoch: 2 ; 	Loss: 0.3298564553260803
# Epoch: 2 ; 	Loss: 0.3747923970222473
# Epoch: 2 ; 	Loss: 0.39405110478401184
# Epoch: 2 ; 	Loss: 0.3992655277252197
# Epoch: 2 ; 	Loss: 0.3789174556732178
# Epoch: 2 ; 	Loss: 0.3495556712150574
# Epoch: 2 ; 	Loss: 0.30129769444465637
# Epoch: 2 ; 	Loss: 0.3600539267063141
# Epoch: 3 ; 	Loss: 0.41171014308929443
# Epoch: 3 ; 	Loss: 0.3562469482421875
# Epoch: 3 ; 	Loss: 0.4012618362903595
# Epoch: 3 ; 	Loss: 0.2834318280220032
# Epoch: 3 ; 	Loss: 0.3406623899936676
# Epoch: 3 ; 	Loss: 0.5299634337425232
# Epoch: 3 ; 	Loss: 0.2672680914402008
# Epoch: 3 ; 	Loss: 0.25573185086250305
# Epoch: 3 ; 	Loss: 0.342937707901001

# Evaluate neural network
with torch.no_grad():  # with no computing gradients for any tensor operation conducted in the inner block
    output = network(x_test)
    test_loss = criterion(output, y_test)
    test_accuracy = (output.round() == y_test).float().mean()
    print("Test Loss:", test_loss.item(), ";", "\tTest Accuracy:", test_accuracy.item())
# Test Loss: 0.25652286410331726 ; 	Test Accuracy: 0.9300000071525574
