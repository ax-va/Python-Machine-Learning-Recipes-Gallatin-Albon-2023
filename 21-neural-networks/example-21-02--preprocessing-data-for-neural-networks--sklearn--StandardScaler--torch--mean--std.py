"""
Preprocess data for use in a neural network.
->
Standardize each feature using Scikit-Learn's StandardScaler.

Since feature values are combined as they pass through individual
NN units (neurons) that are initialized as small random numbers,
it is important that all features have the same scale with
the mean of 0 and the standard deviation of 1.
This is not always necessary; for example, in the case all binary features.

That can be done by using:
1. Scikit-Learn's StandardScaler, or
2. the PyTorch's mean and std methods.
"""
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

# # 1. Scikit-Learn's StandardScaler

# Create feature
features = np.array(
    [[-100.1, 3240.1],
     [-200.2, -234.1],
     [5000.5,  150.1],
     [6000.6, -125.1],
     [9000.9, -673.1]]
)
# array([[-100.1, 3240.1],
#        [-200.2, -234.1],
#        [5000.5,  150.1],
#        [6000.6, -125.1],
#        [9000.9, -673.1]])

# Standardize features
features_std = StandardScaler().fit_transform(features)
# array([[-1.12541308,  1.96429418],
#        [-1.15329466, -0.50068741],
#        [ 0.29529406, -0.22809346],
#        [ 0.57385917, -0.42335076],
#        [ 1.40955451, -0.81216255]])

# Convert to a tensor
features_std_tensor = torch.from_numpy(features_std)
# tensor([[-1.1254,  1.9643],
#         [-1.1533, -0.5007],
#         [ 0.2953, -0.2281],
#         [ 0.5739, -0.4234],
#         [ 1.4096, -0.8122]], dtype=torch.float64)

# # 2. the PyTorch's mean and std methods

# Create features
torch_features = torch.tensor(
    [[-100.1, 3240.1],
          [-200.2, -234.1],
          [5000.5,  150.1],
          [6000.6, -125.1],
          [9000.9, -673.1]],
    requires_grad=True,
)
# tensor([[-100.1000, 3240.1001],
#         [-200.2000, -234.1000],
#         [5000.5000,  150.1000],
#         [6000.6001, -125.1000],
#         [9000.9004, -673.1000]], requires_grad=True)

# Compute the mean and standard deviation
mean = torch_features.mean(0, keepdim=True)
# tensor([[3940.3403,  471.5800]], grad_fn=<MeanBackward1>)
std = torch_features.std(0, unbiased=False, keepdim=True)
# tensor([[3590.1841, 1409.4224]], grad_fn=<StdBackward0>)

# Standardize the features using the mean and standard deviation
torch_features_std = (torch_features - mean) / std
# tensor([[-1.1254,  1.9643],
#         [-1.1533, -0.5007],
#         [ 0.2953, -0.2281],
#         [ 0.5739, -0.4234],
#         [ 1.4096, -0.8122]], grad_fn=<DivBackward0>)
