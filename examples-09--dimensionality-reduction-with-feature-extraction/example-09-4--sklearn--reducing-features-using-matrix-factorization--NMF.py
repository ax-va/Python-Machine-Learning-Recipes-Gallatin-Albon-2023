"""
Reduce the dimensionality of a feature matrix of non-negative values.
->
Use non-negative matrix factorization (NMF).

Constraints:
- only non-negative values
- no explained variance of the outputted features
"""
from sklearn.decomposition import NMF
from sklearn import datasets

# Load the data
digits = datasets.load_digits()
# Load feature matrix
features = digits.data
# array([[ 0.,  0.,  5., ...,  0.,  0.,  0.],
#        [ 0.,  0.,  0., ..., 10.,  0.,  0.],
#        [ 0.,  0.,  0., ..., 16.,  9.,  0.],
#        ...,
#        [ 0.,  0.,  1., ...,  6.,  0.,  0.],
#        [ 0.,  0.,  2., ..., 12.,  0.,  0.],
#        [ 0.,  0., 10., ..., 12.,  1.,  0.]])
features.shape
# (1797, 64)

# Create, fit, and apply NMF
nmf = NMF(n_components=10, random_state=4)
features_nmf = nmf.fit_transform(features)

# Show results
print("Original number of features:", features.shape[1])
# Original number of features: 64
print("Reduced number of features:", features_nmf.shape[1])
# Reduced number of features: 10
