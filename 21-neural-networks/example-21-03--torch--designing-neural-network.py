"""
Design a neural network.
->
Use the PyTorch nn.Module class.

Each unit (neuron) in the hidden layers:
1. Receives a number of inputs.
2. Weights each input by a parameter value.
3. Sums together all weighted inputs along with some bias (typically 0).
4. Most often then applies an *activation function*.
5. Sends the output on to units in the next layer.

The rectified linear unit (ReLU) is a popular activation function in the hidden layers:
$$ relu(z) = max(0, z) $$,
where $z$ is the sum of the weighted inputs and bias.

Output-layers patterns for:
- binary classification -> one unit with a sigmoid activation function;
- multiclass classification -> k units (k = the number of target classes) and a softmax activation function;
- regression -> one unit with no activation function.

Loss functions for:
- binary classification -> binary cross-entropy;
- multiclass classification -> categorical cross-entropy;
- regression -> mean square error (MSE).

Common optimizers:
- stochastic gradient descent,
- stochastic gradient descent with momentum,
- root-mean-square propagation,
- adaptive moment estimation.

Next, we are defining a feedforward two-layer neural network for binary classifications using two ways
- nn.functional,
- the Sequential class

See also:
- PyTorch: Build the Neural Network
https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

- On Loss Functions for Deep Neural Networks in Classification
https://arxiv.org/abs/1702.05659
"""
import torch
import torch.nn as nn


class SimpledNN(nn.Module):
    """
    Feedforward two-layer neural network for binary classifications using nn.functional.
    Each layer is "dense" (also called "fully connected")
    = All the units in the previous layer and in the next layer are connected.
    """
    def __init__(self):
        """ Initiates a network architecture. """
        super().__init__()
        self.fc1 = nn.Linear(10, 16)  # 10 feature values from the observation data
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)  # output layer

    def forward(self, x):
        """ Define activations functions. """
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.sigmoid(self.fc3(x))  # output between 0 and 1
        return x


# Initialize the neural network
feedforward_nn = SimpledNN()

# Define loss function, optimizer
loss_criterion = nn.BCELoss()
optimizer = torch.optim.RMSprop(feedforward_nn.parameters())

feedforward_nn
# SimpledNN(
#   (fc1): Linear(in_features=10, out_features=16, bias=True)
#   (fc2): Linear(in_features=16, out_features=16, bias=True)
#   (fc3): Linear(in_features=16, out_features=1, bias=True)
# )


# another way to define the same neural network
class SequentialNN(nn.Module):
    """
    Feedforward two-layer neural network for binary classifications using the Sequential class.
    Each layer is "dense" (also called "fully connected")
    = All the units in the previous layer and in the next layer are connected.
    """
    def __init__(self):
        """ Initiates a network architecture. """
        super().__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(10, 16),  # 10 feature values from the observation data
            torch.nn.ReLU(),
            torch.nn.Linear(16, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),  # output layer
            torch.nn.Sigmoid()  # output between 0 and 1
        )

    def forward(self, x):
        x = self.sequential(x)
        return x


SequentialNN()
# SequentialNN(
#   (sequential): Sequential(
#     (0): Linear(in_features=10, out_features=16, bias=True)
#     (1): ReLU()
#     (2): Linear(in_features=16, out_features=16, bias=True)
#     (3): ReLU()
#     (4): Linear(in_features=16, out_features=1, bias=True)
#     (5): Sigmoid()
#   )
# )
