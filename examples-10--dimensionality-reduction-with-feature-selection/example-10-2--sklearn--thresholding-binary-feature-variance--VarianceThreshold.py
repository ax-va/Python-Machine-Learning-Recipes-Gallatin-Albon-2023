"""
Filter out binary categorical features with low variance (i.e., likely containing little information).

In binary features (i.e., Bernoulli random variables), variance is calculated as:
Var(x) = p * (1 − p),
where
P(X = x) = 1 - p if x = 0
and
P(X = x) = p if x = 1.
"""
from sklearn.feature_selection import VarianceThreshold

# Create feature matrix with:
# Feature 0: 80% of class 0 and 20% of class 1
# Feature 1: 20% of class 0 and 80% of class 1
# Feature 2: 60% of class 0 and 40% of class 1
features = [[0, 1, 0],
            [0, 1, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0]]

# Three features: feature 0, feature 1, feature 2
# Observations of feature 0: (0, 0, 0, 0, 1) -> 80% of class 0 and 20% of class 1
# Observations of feature 1: (1, 1, 1, 1, 0) -> 20% of class 0 and 80% of class 1
# Observations of feature 2: (0, 1, 0, 1, 0) -> 60% of class 0 and 40% of class 1

# By setting p, remove features where the vast majority of observations are one class
thresholder = VarianceThreshold(threshold=(.75 * (1 - .75)))
features_high_variance = thresholder.fit_transform(features)

# Features 0 and 1 are filtered out, feature 3 remains
features_high_variance.shape
# (5, 1)
features_high_variance
# array([[0],
#        [1],
#        [0],
#        [1],
#        [0]])
thresholder.fit(features).variances_
# array([0.16, 0.16, 0.24])
