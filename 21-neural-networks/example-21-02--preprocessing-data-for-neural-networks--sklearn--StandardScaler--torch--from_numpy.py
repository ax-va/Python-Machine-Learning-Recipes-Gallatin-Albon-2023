"""
Preprocess data for use in a neural network.
->
Standardize each feature using Scikit-Learn's StandardScaler.

Since feature values are combined as they pass through individual
NN units (neurons) that are initialized as small random numbers,
it is important that all features have the same scale.
"""
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

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
