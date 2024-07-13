"""
Filter out a set of numerical features with low variance (i.e., likely containing little information).
->
Use Variance thresholding (VT).

Idea: features with low variance are likely less interesting (and less useful) than features with high variance.

1. Calculate for each feature x the variance:
$Var(x) = 1 / n \sum_{i=1}^n (x_i - x_mean)^2$,
where x_i is an i-th observation of x.

2. Drop all features whose variance does not meet that threshold.

Keep in mind:

1. The variance is not centered.
->
VT will not work when feature sets contain different units (e.g., one feature is in years while another is in dollars).

2. The variance threshold is selected manually.

3. If the features have been standardized (to mean zero and unit variance),
then for obvious reasons VT will not work correctly.
"""
from sklearn import datasets
from sklearn.feature_selection import VarianceThreshold

# Import some data to play with
iris = datasets.load_iris()

# Create features and target
features = iris.data
features.shape
#  (150, 4)
target = iris.target
target.shape
#  (150,)

# Create thresholder
thresholder = VarianceThreshold(threshold=.5)

# Create high variance feature matrix
features_high_variance = thresholder.fit_transform(features)
features_high_variance.shape
# (150, 3)

# View high variance feature matrix
features_high_variance[0:3]
# array([[5.1, 1.4, 0.2],
#        [4.9, 1.4, 0.2],
#        [4.7, 1.3, 0.2]])

# View variances
thresholder.fit(features).variances_
# array([0.68112222, 0.18871289, 3.09550267, 0.57713289])

# Demonstrate the third restriction:
# 3. If the features have been standardized (to mean zero and unit variance),
# then for obvious reasons VT will not work correctly.

# Load library
from sklearn.preprocessing import StandardScaler

# Standardize feature matrix
scaler = StandardScaler()
features_std = scaler.fit_transform(features)

# Calculate variance of each feature
selector = VarianceThreshold()
selector.fit(features_std).variances_
# array([1., 1., 1., 1.])
