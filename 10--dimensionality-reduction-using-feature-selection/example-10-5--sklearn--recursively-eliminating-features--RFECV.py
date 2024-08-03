"""
Automatically select the best features to keep.
->
Use recursive feature elimination (RFE) using cross-validation (CV) (RFECV),
repeatedly training a model, updating the weights or coefficients of that model each time.
The features with the smallest parameters will be removed.

Notice: the features are assumed to be either rescaled or standardized.

The data is split into two groups: a training set and a test set.
The trained model is evaluated on the test set.
After every iteration, cross-validation is used to evaluate the model
to make a decision to stop or to continue.

See also:
- Scikit-Learn: Recursive feature elimination with cross-validation
https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html#sphx-glr-auto-examples-feature-selection-plot-rfe-with-cross-validation-py
"""
import warnings
from sklearn.datasets import make_regression
from sklearn.feature_selection import RFECV
from sklearn import datasets, linear_model

# Suppress an annoying but harmless warning
warnings.filterwarnings(
    action="ignore",
    module="scipy",
    message="^internal gelsd",
)

# Generate features matrix, target vector, and the true coefficients
features, target = make_regression(
    n_samples=10000,
    n_features=100,
    n_informative=2,
    random_state=1,
)
features
# array([[-6.42807348e-01, -7.61659694e-01, -2.11015976e-01,
#          1.63056033e-01,  6.42126127e-01,  8.50798578e-03,
#         -3.90428801e-01,  1.13220029e+00,  7.27660535e-01,
#         -9.18046613e-01, -1.72570086e+00,  1.74709864e+00,
#          3.20767102e-01,  5.13428565e-01, -1.32562545e+00,
# ...
features.shape
# (10000, 100)
target
# array([  27.46715058,   23.8681199 ,   27.34686337, ..., -116.75887199,
#         -24.22656921,   -6.46891152])
target.shape
# (10000,)

# Create a linear regression
linear_regression = linear_model.LinearRegression()

# Recursively eliminate features
rfecv = RFECV(
    estimator=linear_regression,  # type of model to train
    step=1,  # number or proportion of features to drop during each loop
    scoring="neg_mean_squared_error",  # metric of quality during cross-validation
)
rfecv.fit(features, target)
# RFECV(estimator=LinearRegression(), scoring='neg_mean_squared_error')
rfecv.transform(features)
# array([[ 0.00850799,  0.009944  ,  0.7031277 ],
#        [-1.07500204,  0.04996652,  2.56148527],
#        [ 1.37940721, -0.55102204, -1.77039484],
#        ...,
#        [-0.80331656,  1.22981262, -1.60648007],
#        [ 0.39508844, -1.14347573, -1.34564911],
#        [-0.55383035, -1.39387381,  0.82880112]])

# Number of best features
rfecv.n_features_
# 3

# See which of those features we should keep: only three of hundred marked as True
rfecv.support_
# array([False, False, False, False, False,  True, False, False, False,
#        False, False, False, False, False, False,  True, False, False,
#        False, False, False, False, False, False, False, False, False,
#        False, False, False, False, False, False, False, False, False,
#        False, False, False,  True, False, False, False, False, False,
#        False, False, False, False, False, False, False, False, False,
#        False, False, False, False, False, False, False, False, False,
#        False, False, False, False, False, False, False, False, False,
#        False, False, False, False, False, False, False, False, False,
#        False, False, False, False, False, False, False, False, False,
#        False, False, False, False, False, False, False, False, False,
#        False])

# View the rankings of the features: best (1) to worst
rfecv.ranking_
# array([47, 96, 46, 67,  8,  1, 11,  5, 23, 21, 31, 14, 27, 66, 18,  1, 29,
#        98, 20, 58, 85, 44, 13, 24,  6,  9, 35, 88,  3, 73, 60, 56, 52, 32,
#        78, 72, 51, 15, 41,  1, 12, 16, 28, 71, 70, 63, 86, 25, 17, 97, 48,
#        10, 95, 90, 83, 93, 89, 61, 75, 26, 37, 22, 87, 30, 45, 82,  2, 43,
#        39, 33, 38, 40, 81,  4, 57,  7, 64, 34, 53, 91, 77, 79, 19, 36, 42,
#        49, 50, 54, 68, 55, 69, 74, 62, 76, 80, 92, 84, 65, 59, 94])
