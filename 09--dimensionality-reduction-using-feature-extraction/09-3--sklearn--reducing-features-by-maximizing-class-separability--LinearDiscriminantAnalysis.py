"""
Reduce the number of features to be used by a classifier by maximizing the separation between the classes.
->
Apply linear discriminant analysis (LDA) to project the features
onto component axes that maximize the separation of classes.

In PCA: interested only in the component axes that maximize the variance in the data.
In LDA: the additional goal of maximizing the differences between classes.

See also:
- Scikit-Learn: Comparison of LDA and PCA 2D projection of Iris dataset
https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_lda.html#sphx-glr-auto-examples-decomposition-plot-pca-vs-lda-py

- Sebastian Raschka: Linear Discriminant Analysis
https://sebastianraschka.com/Articles/2014_python_lda.html
"""
from typing import Iterable
import numpy as np
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Load Iris flower dataset:
iris = datasets.load_iris()
# {'data': array([[5.1, 3.5, 1.4, 0.2],
#         [4.9, 3. , 1.4, 0.2],
#         [4.7, 3.2, 1.3, 0.2],
#         [4.6, 3.1, 1.5, 0.2],
#         [5. , 3.6, 1.4, 0.2],
# ...
#  'target': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]),
#  'frame': None,
#  'target_names': array(['setosa', 'versicolor', 'virginica'], dtype='<U10'),
# ...
features = iris.data
features.shape
# (150, 4)
target = iris.target
target.shape
# (150,)

# Create and run an LDA, then use it to transform the features
lda = LinearDiscriminantAnalysis(n_components=1)
features_lda = lda.fit(features, target).transform(features)
# array([[ 8.06179978],
#        [ 7.12868772],
#        [ 7.48982797],
#        [ 6.81320057],
#        [ 8.13230933],
# ...
features_lda.shape
# (150, 1)

# Print the number of features
print("Original number of features:", features.shape[1])
# Original number of features: 4
print("Reduced number of features:", features_lda.shape[1])
# Reduced number of features: 1

# View the amount of variance explained by each component
lda.explained_variance_ratio_
# array([0.9912126])

# Get the ratio of variance explained by every component feature
lda = LinearDiscriminantAnalysis(n_components=None)
features_lda = lda.fit(features, target)
# LinearDiscriminantAnalysis()
lda_var_ratios = lda.explained_variance_ratio_
# array([0.9912126, 0.0087874])
features_lda.transform(features)
# array([[ 8.06179978e+00, -3.00420621e-01],
#        [ 7.12868772e+00,  7.86660426e-01],
#        [ 7.48982797e+00,  2.65384488e-01],
#        [ 6.81320057e+00,  6.70631068e-01],
#        [ 8.13230933e+00, -5.14462530e-01],
# ...


def select_n_components_v1(var_ratios: Iterable, var_threshold: float) -> int:
    """
    Selects the number of components passing through the threshold of variance for LDA.
    Args:
        var_ratios: descending variance ratios
        var_threshold: accumulated variance threshold
    Returns:
        n_components
    """
    total_variance = 0.0
    # Set initial number of features
    n_components = 0
    # For the explained variance of each feature:
    for explained_variance in var_ratios:
        # Add the explained variance to the total
        total_variance += explained_variance
        # Add one to the number of components
        n_components += 1
        # If we reach our goal level of explained variance
        if total_variance >= var_threshold:
            # End the loop
            break
    # Return the number of components
    return n_components


select_n_components_v1(lda_var_ratios, 0.95)
# 1

np.add.accumulate(lda_var_ratios)
# array([0.9912126, 1.       ])


def select_n_components_v2(var_ratios: np.array, var_threshold: float) -> int:
    """
    Selects the number of components passing through the threshold of variance for LDA.
    Args:
        var_ratios: descending variance ratios
        var_threshold: accumulated variance threshold
    Returns:
        n_components
    """
    acc = np.add.accumulate(var_ratios)
    n_components = 1 + var_ratios[acc < var_threshold].size
    return n_components


select_n_components_v2(lda_var_ratios, 0.95)
# 1
