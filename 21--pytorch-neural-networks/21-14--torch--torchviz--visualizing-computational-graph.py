"""
Visualize the computational graph of a PyTorch neural network.
->
Use the torchviz package.
"""
import torch
import torch.nn as nn
from torchviz import make_dot

# Set random seed for PyTorch
torch.manual_seed(0)


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
# Create input
input_features = torch.randn(1, 10)
# Get output
output = network(input_features)

# Visualize the computational graph
make_dot(
    output,
    params=dict(network.named_parameters())
).render(
    "example-21-14--torch--torchviz--visualizing-computational-graph",
    format="svg",
)
