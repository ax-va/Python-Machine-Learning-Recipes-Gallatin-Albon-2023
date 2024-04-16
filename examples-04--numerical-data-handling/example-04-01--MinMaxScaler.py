"""
Rescaling the values of numerical features to be between two values.
Min-max scaling uses the following formula:
z_i = (x_i - x_min) / (x_max - x_min)
"""
import numpy as np
from sklearn import preprocessing

# Create features
features = np.array([[-500.5],
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
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

# Scale features
features_scaled = scaler.fit_transform(features)
# array([[0.        ],
#        [0.28571429],
#        [0.35714286],
#        [0.42857143],
#        [1.        ]])

# Alternatively
scaler.fit(features)
# MinMaxScaler()
scaler.transform(features)
# array([[0.        ],
#        [0.28571429],
#        [0.35714286],
#        [0.42857143],
#        [1.        ]])
