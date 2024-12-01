"""
Save a PyTorch model and load it elsewhere.
->
Use the torch.save and torch.load functions.

Re-initialize the model and load the model parameters using `load_state_dict`.

See also:
- PyTorch: Saving and Loading Models
https://pytorch.org/tutorials/beginner/saving_loading_models.html
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import RMSprop
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

NUM_EPOCHS = 5

torch_version = torch.__version__.replace(".", "_")

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
for epoch_idx in range(1, NUM_EPOCHS + 1):  # how many epochs to use when training the data
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)  # using the forward method
        loss = criterion(output, target)
        loss.backward()  # to update the gradients
        optimizer.step()
    print("Epoch:", epoch_idx, ";", "\tLoss:", loss.item())
    # Epoch: 1 ; 	Loss: 0.42521628737449646
    # Epoch: 2 ; 	Loss: 0.3600539267063141
    # Epoch: 3 ; 	Loss: 0.342937707901001
    # Epoch: 4 ; 	Loss: 0.3029881417751312
    # Epoch: 5 ; 	Loss: 0.38423997163772583

# Save the model after it's been trained
torch.save(
    {
        'epoch': NUM_EPOCHS,
        'model_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    },
    f"models/model_torch_{torch_version}.pt"
)

# Reinitialize neural network
network = SequentialNN()
state_dict = torch.load(
    f"models/model_torch_{torch_version}.pt",
    map_location=torch.device('cpu')
)["model_state_dict"]
network.load_state_dict(state_dict, strict=False)
network.eval()
