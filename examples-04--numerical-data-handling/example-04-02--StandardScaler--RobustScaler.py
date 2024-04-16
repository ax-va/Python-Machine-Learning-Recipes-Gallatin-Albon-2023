"""
Transform feature values to have a mean of 0 and a standard deviation of 1.

This scaling uses the following formula:
z_i = (x_i - x_mean) / sigma_x.

For example, PCA often works better using standardization,
while min-max scaling is often recommended for neural networks.

See also a z-score in statistics.
"""
import numpy as np
from sklearn import preprocessing

# Create feature
x = np.array([[-1000.1],
              [-200.2],
              [500.5],
              [600.6],
              [9000.9]])
# array([[-1000.1],
#        [ -200.2],
#        [  500.5],
#        [  600.6],
#        [ 9000.9]])

# Create scaler
scaler = preprocessing.StandardScaler()

# Transform the feature
standardized = scaler.fit_transform(x)
# array([[-0.76058269],
#        [-0.54177196],
#        [-0.35009716],
#        [-0.32271504],
#        [ 1.97516685]])

print("Mean:", standardized.mean())
# Mean: 4.4408920985006264e-17
print("Mean rounded:", round(standardized.mean()))
# Mean rounded: 0
print("Standard deviation:", standardized.std())
# Standard deviation: 1.0

# If data has significant outliers, RobustScaler is recommended
robust_scaler = preprocessing.RobustScaler()

# Transform features
robust_scaler.fit_transform(x)
# array([[-1.87387612],
#        [-0.875     ],
#        [ 0.        ],
#        [ 0.125     ],
#        [10.61488511]])
