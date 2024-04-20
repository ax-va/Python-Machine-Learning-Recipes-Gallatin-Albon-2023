"""
Create polynomial and interaction features.
"""
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# Create feature matrix: each row is an observation of the features x_1 and x_2
features = np.array([[2, 3],
                     [2, 3],
                     [2, 3]])
# array([[2, 3],
#        [2, 3],
#        [2, 3]])

# Create PolynomialFeatures instance
polynomial_features = PolynomialFeatures(degree=2, include_bias=False)
# PolynomialFeatures(include_bias=False)

# Here, degree=2 creates new features up to the second power with the interaction feature:
# x_1, x_2, x_1^2, x1 * x2, x_2^2

# Create polynomial features
polynomial_features.fit_transform(features)
# array([[2., 3., 4., 6., 9.],
#        [2., 3., 4., 6., 9.],
#        [2., 3., 4., 6., 9.]])

# Restrict the new features created to only interaction features.
# Here: x_1, x_2, x_1 * x_2
interaction = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
# PolynomialFeatures(include_bias=False, interaction_only=True)

interaction.fit_transform(features)
# array([[2., 3., 6.],
#        [2., 3., 6.],
#        [2., 3., 6.]])
