"""
Reduce overfitting by regularizing the weights of your network.
->
Penalize the parameters of the network, also called *weight regularization*.

Notice:
python-dev for using Python API for C should be installed;
if not, install it on Ubuntu
$ sudo apt-get install python3.x-dev
where 3.x is your Python version in your virtual environment.
"""
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

NUM_EPOCHS = 1000

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


# Create neural network
network = SequentialNN()

# Define loss function, optimizer
criterion = nn.BCELoss()
optimizer = Adam(
    network.parameters(),
    lr=1e-4,
    # weight_decay determines how much to penalize higher parameter values
    weight_decay=1e-5,
    # Values greater than 0 indicate L2 regularization in PyTorch
)

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
train_losses = []
test_losses = []
for epoch_idx in range(1, NUM_EPOCHS + 1):  # how many epochs to use when training the data
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)  # using the forward method
        loss = criterion(output, target)
        loss.backward()  # to update the gradients
        optimizer.step()
    print("Epoch:", epoch_idx, ";", "\tLoss:", loss.item())
    # Epoch: 1 ; 	Loss: 0.7038822770118713
    # Epoch: 2 ; 	Loss: 0.6934329867362976
    # Epoch: 3 ; 	Loss: 0.6797105669975281
    # ...
    # Epoch: 998 ; 	Loss: 0.4011973440647125
    # Epoch: 999 ; 	Loss: 0.3576684594154358
    # Epoch: 1000 ; Loss: 0.27274689078330994

    with torch.no_grad():
        train_output = network(x_train)
        train_loss = criterion(train_output, y_train)
        train_losses.append(train_loss.item())
        test_output = network(x_test)
        test_loss = criterion(test_output, y_test)
        test_losses.append(test_loss.item())

# Evaluate neural network
with torch.no_grad():  # with no computing gradients for any tensor operation conducted in the inner block
    output = network(x_test)
    test_loss = criterion(output, y_test)
    test_accuracy = (output.round() == y_test).float().mean()
    print("Test Loss:", test_loss.item(), ";",
          "\tTest Accuracy:", test_accuracy.item())
    # Test Loss: 0.2374035269021988 ; 	Test Accuracy: 0.9399999976158142


# Visualize loss history for 100 epochs
num_epochs_less = NUM_EPOCHS // 10
epochs = range(1, num_epochs_less+1)
plt.plot(epochs, train_losses[:num_epochs_less], "r--")
plt.plot(epochs, test_losses[:num_epochs_less], "b-")
plt.legend(["Training Loss", "Test Loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
# plt.show()
plt.savefig('example-21-09-1--torch--reducing-overfitting-with-weight-regularization--Adam.svg')
plt.close()
# Visualize loss history for 1000 epochs
epochs = range(1, NUM_EPOCHS+1)
plt.plot(epochs, train_losses, "r--")
plt.plot(epochs, test_losses, "b-")
plt.legend(["Training Loss", "Test Loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
# plt.show()
plt.savefig('example-21-09-2--torch--reducing-overfitting-with-weight-regularization--Adam.svg')
plt.close()
