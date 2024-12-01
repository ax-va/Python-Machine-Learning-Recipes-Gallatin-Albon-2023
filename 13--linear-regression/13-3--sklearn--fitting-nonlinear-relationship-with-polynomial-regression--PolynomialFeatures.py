"""
Model a nonlinear relationship.
"""
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import make_regression

# Generate features matrix, target vector
features, target = make_regression(
    n_samples=100,
    n_features=3,
    n_informative=2,
    n_targets=1,
    noise=0.2,
    coef=False,
    random_state=1,
)

features.shape
# (100, 3)
features[0]
# array([ 0.58591043,  0.78477065, -0.95542526])

# Create polynomial features x^2 and x^3
polynomial = PolynomialFeatures(degree=3, include_bias=False)
features_polynomial = polynomial.fit_transform(features)
features_polynomial.shape
# (100, 19)
features_polynomial[0]
# array([ 0.58591043,  0.78477065, -0.95542526,  0.34329103,  0.45980531,
#        -0.55979363,  0.61586497, -0.74978971,  0.91283743,  0.2011378 ,
#         0.26940473, -0.32798893,  0.36084171, -0.43930961,  0.53484097,
#         0.48331276, -0.58841296,  0.71636803, -0.87214794])

# x_1, x_2, x_3, x_1², x_1 * x_2,
# x_1 * x_3, x_2², x_2 * x_3, x_3², x_1³,
# x_1² * x_2, x_1² * x_3, x_1 * x_2², x_1 * x_2 * x_3, x_1 * x_3²,
# x_2³, x_2² * x_3, x_2 * x_3², x_2 * x_3², x_3³

# Create linear regression
regression = LinearRegression()
# Fit the linear regression with non-linear terms
model = regression.fit(features_polynomial, target)
