"""
Train a neural network for regression.
->
Construct a feedforward neural network with an output layer
with no activation function in PyTorch and train it.

Notice:
python-dev for using Python API for C should be installed;
if not, install it on Ubuntu
$ sudo apt-get install python3.x-dev
where 3.x is your Python version in your virtual environment.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import RMSprop
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

NUM_EPOCHS = 5

# Create data with 10 features and 1000 observations
features, target = make_regression(
    n_features=10,
    n_samples=1000,
    random_state=1,
)
# Because we are using simulated data using Scikit-Learn make_regression,
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

# Set random seeds for PyTorch
torch.manual_seed(0)

# Convert data to PyTorch tensors
x_train = torch.from_numpy(features_train).float()
y_train = torch.from_numpy(target_train).float().view(-1,1)
x_test = torch.from_numpy(features_test).float()
y_test = torch.from_numpy(target_test).float().view(-1,1)


# Define a neural network using "Sequential"
class SequentialNN(nn.Module):
    """
    Feedforward two-layer neural network for regression using nn.Sequential.
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
            torch.nn.Linear(16, 1),  # 1 unit (neuron) in the output layer
            # no output activation function
        )

    def forward(self, x):
        x = self.sequential(x)
        return x


# Create neural network
network = SequentialNN()

# Define loss function, optimizer
criterion = nn.MSELoss()  # mean-square-error loss function
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
        # Epoch: 1 ; 	Loss: 28634.373046875
        # Epoch: 1 ; 	Loss: 30790.064453125
        # Epoch: 1 ; 	Loss: 28336.96484375
        # Epoch: 1 ; 	Loss: 29356.412109375
        # Epoch: 1 ; 	Loss: 31970.08203125
        # Epoch: 1 ; 	Loss: 32253.671875
        # Epoch: 1 ; 	Loss: 26847.642578125
        # Epoch: 1 ; 	Loss: 21875.431640625
        # Epoch: 1 ; 	Loss: 24668.900390625
        # Epoch: 2 ; 	Loss: 20348.119140625
        # Epoch: 2 ; 	Loss: 14060.6328125
        # Epoch: 2 ; 	Loss: 15267.744140625
        # Epoch: 2 ; 	Loss: 8771.056640625
        # Epoch: 2 ; 	Loss: 7012.18896484375
        # Epoch: 2 ; 	Loss: 7827.31640625
        # Epoch: 2 ; 	Loss: 6225.88134765625
        # Epoch: 2 ; 	Loss: 5441.62744140625
        # Epoch: 2 ; 	Loss: 3907.996826171875
        # Epoch: 3 ; 	Loss: 3447.470703125
        # Epoch: 3 ; 	Loss: 3383.4775390625
        # Epoch: 3 ; 	Loss: 2714.465576171875
        # Epoch: 3 ; 	Loss: 1763.8612060546875
        # Epoch: 3 ; 	Loss: 1396.1829833984375
        # Epoch: 3 ; 	Loss: 1501.1131591796875
        # Epoch: 3 ; 	Loss: 2090.01708984375
        # Epoch: 3 ; 	Loss: 784.6046142578125
        # Epoch: 3 ; 	Loss: 558.103759765625
        # Epoch: 4 ; 	Loss: 828.4906005859375
        # Epoch: 4 ; 	Loss: 729.664794921875
        # Epoch: 4 ; 	Loss: 959.2522583007812
        # Epoch: 4 ; 	Loss: 586.7230224609375
        # Epoch: 4 ; 	Loss: 468.0014953613281
        # Epoch: 4 ; 	Loss: 862.5960693359375
        # Epoch: 4 ; 	Loss: 466.1515197753906
        # Epoch: 4 ; 	Loss: 488.13983154296875
        # Epoch: 4 ; 	Loss: 512.484130859375
        # Epoch: 5 ; 	Loss: 342.1610107421875
        # Epoch: 5 ; 	Loss: 521.171630859375
        # Epoch: 5 ; 	Loss: 379.0733337402344
        # Epoch: 5 ; 	Loss: 501.30426025390625
        # Epoch: 5 ; 	Loss: 413.69354248046875
        # Epoch: 5 ; 	Loss: 373.8358459472656
        # Epoch: 5 ; 	Loss: 455.057373046875
        # Epoch: 5 ; 	Loss: 345.7730712890625
        # Epoch: 5 ; 	Loss: 316.62969970703125

# Evaluate neural network
with torch.no_grad():  # with no computing gradients for any tensor operation conducted in the inner block
    output = network(x_test)
    test_loss = float(criterion(output, y_test))
    print("Test MSE:", test_loss)
    # Test MSE: 235.0173797607422
