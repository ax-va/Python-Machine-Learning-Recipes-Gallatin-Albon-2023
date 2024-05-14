"""
Rescale the vector to be between two values.

Min-max scaling uses the following formula:
z_i = (x_i - x_min) / (x_max - x_min).

For example, PCA often works better using standardization,
while min-max scaling is often recommended for neural networks.
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Create vector
x = np.array([[-500.5],
              [-100.1],
              [0],
              [100.1],
              [900.9]])
# array([[-500.5],
#        [-100.1],
#        [   0. ],
#        [ 100.1],
#        [ 900.9]])

# Create scaler
minmax_scaler = MinMaxScaler(feature_range=(0, 1))

# Scale vector
x_scaled = minmax_scaler.fit_transform(x)
# array([[0.        ],
#        [0.28571429],
#        [0.35714286],
#        [0.42857143],
#        [1.        ]])

# Alternatively
minmax_scaler.fit(x)
# MinMaxScaler()
minmax_scaler.transform(x)
# array([[0.        ],
#        [0.28571429],
#        [0.35714286],
#        [0.42857143],
#        [1.        ]])
