"""
Have a feature whose effect on the target variable depends on another feature.

Relationship between a target and only three features is represented as follows:
y = bias + beta_1 * feature_1 + beta_2 * feature_2 + beta_3 * feature_1 * feature_2 + error

If we believe there is an interaction between features, we can use
PolynomialFeatures to create interaction terms for all combinations of features.
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import make_regression

# Generate features matrix, target vector
features, target = make_regression(
    n_samples=100,
    n_features=2,
    n_informative=2,
    n_targets=1,
    noise=0.2,
    coef=False,
    random_state=1,
)

features.shape
# (100, 2)

target.shape
# (100,)

# Create interaction term
interaction = PolynomialFeatures(
    degree=3,  # in this case the same as degree=2
    include_bias=False,
    interaction_only=True,  # Return only interaction terms
)
features_interaction = interaction.fit_transform(features)
features_interaction.shape
# (100, 3)

# View the feature values for first observation
features[0]
# array([0.0465673 , 0.80186103])

features_interaction[0]
# array([0.0465673 , 0.80186103, 0.0373405 ])
# 0.0465673 *  0.80186103 = 0.037340503142319

# another example
other_features = np.array([[2, 3],
                           [2, 3],
                           [2, 3]])
other_features_interaction = interaction.fit_transform(other_features)
# array([[2., 3., 6.],
#        [2., 3., 6.],
#        [2., 3., 6.]])

# Here, we have
# x_1, x_2, x1 * x2

# For each observation, multiply the values of the first and second feature
interaction_term = features[:, 0] * features[:, 1]
interaction_term.shape
# (100,)

interaction_term[0]
# 0.037340501965846186

# Use linear regression with interaction terms
regression = LinearRegression()
# Fit the linear regression
model = regression.fit(features_interaction, target)
