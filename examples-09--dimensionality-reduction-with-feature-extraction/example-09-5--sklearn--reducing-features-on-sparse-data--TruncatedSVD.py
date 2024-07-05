"""
Reduce the dimensionality of a sparse feature matrix.
->
Use Truncated Singular Value Decomposition (TSVD).

The practical advantage of TSVD is that, unlike PCA, it works on sparse feature matrices.

It uses a random number generator.
->
The signs of the output can flip between fittings.
->
Use the fit method only once.

See also:
- Scikit-Learn: TruncatedSVD
https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from sklearn import datasets

# Load the data
digits = datasets.load_digits()
digits.data
# array([[ 0.,  0.,  5., ...,  0.,  0.,  0.],
#        [ 0.,  0.,  0., ..., 10.,  0.,  0.],
#        [ 0.,  0.,  0., ..., 16.,  9.,  0.],
#        ...,
#        [ 0.,  0.,  1., ...,  6.,  0.,  0.],
#        [ 0.,  0.,  2., ..., 12.,  0.,  0.],
#        [ 0.,  0., 10., ..., 12.,  1.,  0.]])
digits.data.shape
# (1797, 64)

# Standardize feature matrix
features = StandardScaler().fit_transform(digits.data)

# Make sparse matrix
features_sparse = csr_matrix(features)

# Create a TSVD
tsvd = TruncatedSVD(n_components=10)

# Conduct TSVD on sparse matrix
features_sparse_tsvd = tsvd.fit(features_sparse).transform(features_sparse)

# Show results
print("Original number of features:", features_sparse.shape[1])
# Original number of features: 64
print("Reduced number of features:", features_sparse_tsvd.shape[1])
# Reduced number of features: 10

# Sum of first three components' explained variance ratios
tsvd.explained_variance_ratio_[0:3].sum()
# 0.30039385392472984

# Create and run a TSVD with one less than number of features
tsvd = TruncatedSVD(n_components=features_sparse.shape[1]-1)
features_tsvd = tsvd.fit(features)

# # # Find n_components for given variance

# List of explained variances
tsvd_var_ratios = tsvd.explained_variance_ratio_
# array([1.20339161e-01, 9.56105440e-02, 8.44441489e-02, 6.49840791e-02,
#        4.86015488e-02, 4.21411987e-02, 3.94208280e-02, 3.38938092e-02,
#        2.99822101e-02, 2.93200255e-02, 2.78180546e-02, 2.57705509e-02,
#        2.27530332e-02, 2.22717974e-02, 2.16522943e-02, 1.91416661e-02,
#        1.77554709e-02, 1.63806927e-02, 1.59646017e-02, 1.48919119e-02,
#        1.34796957e-02, 1.27193137e-02, 1.16583735e-02, 1.05764660e-02,
#        9.75315947e-03, 9.44558990e-03, 8.63013827e-03, 8.36642854e-03,
#        7.97693248e-03, 7.46471371e-03, 7.25582151e-03, 6.91911245e-03,
#        6.53908536e-03, 6.40792574e-03, 5.91384112e-03, 5.71162405e-03,
#        5.23636803e-03, 4.81807586e-03, 4.53719260e-03, 4.23162753e-03,
#        4.06053070e-03, 3.97084808e-03, 3.56493303e-03, 3.40787181e-03,
#        3.27835335e-03, 3.11032007e-03, 2.88575294e-03, 2.76489264e-03,
#        2.59174941e-03, 2.34483006e-03, 2.18256858e-03, 2.03597635e-03,
#        1.95512426e-03, 1.83318499e-03, 1.67946387e-03, 1.61236062e-03,
#        1.47762694e-03, 1.35118411e-03, 1.25100742e-03, 1.03695730e-03,
#        8.25350945e-04, 4.04744788e-33, 5.08670952e-33])


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

select_n_components_v2(tsvd_var_ratios, 0.95)
# 40
