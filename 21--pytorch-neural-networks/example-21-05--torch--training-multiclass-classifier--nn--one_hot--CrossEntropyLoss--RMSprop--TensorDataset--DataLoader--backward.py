"""
Train a multiclass classifier neural network.
->
Construct a feedforward neural network with an output layer with
using the softmax activation function in PyTorch and train it.

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

NUM_CLASSES = 3
NUM_EPOCHS = 3

# Create data with 10 features and 1000 observations
features, target = make_classification(
    n_classes=NUM_CLASSES,
    n_informative=9,
    n_redundant=0,
    n_features=10,
    n_samples=1000,
)
# Split training and test sets
features_train, features_test, target_train, target_test = train_test_split(
    features, target,
    test_size=0.1,
    random_state=1,
)

# Set random seeds
torch.manual_seed(0)
np.random.seed(0)

# Convert data to PyTorch tensors
x_train = torch.from_numpy(features_train).float()
y_train = torch.nn.functional.one_hot(
    torch.from_numpy(target_train).long(),
    num_classes=NUM_CLASSES,
).float()
x_test = torch.from_numpy(features_test).float()
y_test = torch.nn.functional.one_hot(  # one-hot encoded array
    torch.from_numpy(target_test).long(),
    num_classes=NUM_CLASSES,
).float()


# Define a neural network using "Sequential"
class SequentialNN(nn.Module):
    """
    Feedforward two-layer neural network for multiclass classification using nn.Sequential.
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
            torch.nn.Linear(16, NUM_CLASSES),  # 3 units (neurons) in the output layer
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.sequential(x)
        return x


# Create neural network
network = SequentialNN()

# Define loss function, optimizer.
# The target must be one-hot encoded to use
# the categorical cross-entropy loss function.
criterion = nn.CrossEntropyLoss()
optimizer = RMSprop(network.parameters())

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

# Compile the model using torch 2.0's optimizer
network = torch.compile(network)

# Train neural network
for epoch_idx in range(NUM_EPOCHS):  # how many epochs to use when training the data
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)  # using the forward method
        loss = criterion(output, target)
        loss.backward()  # to update the gradients
        optimizer.step()
        print("Epoch:", epoch_idx + 1, ";", "\tLoss:", loss.item())
        # Epoch: 1 ; 	Loss: 1.102777361869812
        # Epoch: 1 ; 	Loss: 1.0140036344528198
        # Epoch: 1 ; 	Loss: 0.9002036452293396
        # Epoch: 1 ; 	Loss: 0.8664718866348267
        # Epoch: 1 ; 	Loss: 0.8068368434906006
        # Epoch: 1 ; 	Loss: 0.8978130221366882
        # Epoch: 1 ; 	Loss: 0.845928430557251
        # Epoch: 1 ; 	Loss: 0.8498579263687134
        # Epoch: 1 ; 	Loss: 0.7600250840187073
        # Epoch: 2 ; 	Loss: 0.775653064250946
        # Epoch: 2 ; 	Loss: 0.7758886814117432
        # Epoch: 2 ; 	Loss: 0.8230465054512024
        # Epoch: 2 ; 	Loss: 0.8089917898178101
        # Epoch: 2 ; 	Loss: 0.7518138885498047
        # Epoch: 2 ; 	Loss: 0.7131175994873047
        # Epoch: 2 ; 	Loss: 0.7558932304382324
        # Epoch: 2 ; 	Loss: 0.7803115844726562
        # Epoch: 2 ; 	Loss: 0.7786457538604736
        # Epoch: 3 ; 	Loss: 0.7506158351898193
        # Epoch: 3 ; 	Loss: 0.7889423370361328
        # Epoch: 3 ; 	Loss: 0.7288426160812378
        # Epoch: 3 ; 	Loss: 0.7012325525283813
        # Epoch: 3 ; 	Loss: 0.6914394497871399
        # Epoch: 3 ; 	Loss: 0.7751169800758362
        # Epoch: 3 ; 	Loss: 0.6826244592666626
        # Epoch: 3 ; 	Loss: 0.7325385212898254
        # Epoch: 3 ; 	Loss: 0.7058666944503784

# Evaluate neural network
with torch.no_grad():  # with no computing gradients for any tensor operation conducted in the inner block
    output = network(x_test)
    test_loss = criterion(output, y_test)
    test_accuracy = (output.round() == y_test).float().mean()
    print("Test Loss:", test_loss.item(), ";",
          "\tTest Accuracy:", test_accuracy.item())
    # Test Loss: 0.7027209997177124 ; 	Test Accuracy: 0.8933333158493042
