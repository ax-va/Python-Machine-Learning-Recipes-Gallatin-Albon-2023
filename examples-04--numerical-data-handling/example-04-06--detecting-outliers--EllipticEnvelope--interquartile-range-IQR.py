"""
Collect techniques to identify extreme observations (outliers):
- EllipticEnvelope looking at all the observations -> the value of contamination is needed to know
- interquartile range (IQR) for an individual feature's observation
"""
from typing import Tuple

import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.datasets import make_blobs

# Create simulated data
features, target = make_blobs(
    n_samples=10,
    n_features=2,  # number of features: x_1 nad x_2
    centers=1,  # number of clusters
    random_state=1,
)

features
# array([[-1.83198811,  3.52863145],
#        [-2.76017908,  5.55121358],
#        [-1.61734616,  4.98930508],
#        [-0.52579046,  3.3065986 ],
#        [ 0.08525186,  3.64528297],
#        [-0.79415228,  2.10495117],
#        [-1.34052081,  4.15711949],
#        [-1.98197711,  4.02243551],
#        [-2.18773166,  3.33352125],
#        [-0.19745197,  2.34634916]])

target
# array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# Replace the first row with extreme values
features[0, 0] = 10000
features[0, 1] = 10000
features
# array([[ 1.00000000e+04,  1.00000000e+04],
#        [-2.76017908e+00,  5.55121358e+00],
#        [-1.61734616e+00,  4.98930508e+00],
#        [-5.25790464e-01,  3.30659860e+00],
#        [ 8.52518583e-02,  3.64528297e+00],
#        [-7.94152277e-01,  2.10495117e+00],
#        [-1.34052081e+00,  4.15711949e+00],
#        [-1.98197711e+00,  4.02243551e+00],
#        [-2.18773166e+00,  3.33352125e+00],
#        [-1.97451969e-01,  2.34634916e+00]])

# Create detector
outlier_detector = EllipticEnvelope(
    contamination=.1  # proportion of observations that are outliers
)

# Fit detector
outlier_detector.fit(features)

# Predict outliers
outlier_detector.predict(features)
# array([-1,  1,  1,  1,  1,  1,  1,  1,  1,  1])

# Values of -1 refer to outliers whereas values of 1 refer to inliers

# Another way:
# Look at individual features and identify extreme values in those features using interquartile range (IQR)

# Separate one feature x_1
feature_1 = features[:, 0]
# array([ 1.00000000e+04, -2.76017908e+00, -1.61734616e+00, -5.25790464e-01,
#         8.52518583e-02, -7.94152277e-01, -1.34052081e+00, -1.98197711e+00,
#        -2.18773166e+00, -1.97451969e-01])

q1, q3 = np.percentile(feature_1, [25, 75])
q1
# -1.890819372279752
q3
# -0.2795365925809296
iqr = q3 - q1
# 1.6112827796988223
1.5 * iqr
# 2.4169241695482335
lower_bound = q1 - 1.5 * iqr
# -4.3077435418279855
upper_bound = q3 + 1.5 * iqr
# 2.137387576967304

feature_1[1] = 2000  # another outlier
feature_1
# array([ 1.00000000e+04,  2.00000000e+03, -1.61734616e+00, -5.25790464e-01,
#         8.52518583e-02, -7.94152277e-01, -1.34052081e+00, -1.98197711e+00,
#        -2.18773166e+00, -1.97451969e-01])
q1, q3 = np.percentile(feature_1, [25, 75])
q1
# -1.5481398218975504
q3
# 0.014575901475083974
iqr = q3 - q1
# 1.5627157233726343
lower_bound = q1 - 1.5 * iqr
# -3.8922134069565018
upper_bound = q3 + 1.5 * iqr
# 2.3586494865340355


# Create a function to return index of outliers
def outlier_indices(x: np.array) -> np.array:
    """
    Returns indices of outliers using the interquartile range (IQR).
    IQR is the difference between the third and first quartiles of a vector.
    Outliers typically represent values 1.5*IQR below the first quartile
    and 1.5*IQR above the third quartile.

    Args:
        x: vector of observations
    Returns:
        indices of outliers
    """
    q1, q3 = np.percentile(x, [25, 75])
    # Calculate IQR
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    return np.where((x < lower_bound) | (x > upper_bound))[0]


# Detect two outliers
outlier_indices(feature_1)
# array([0, 1])
