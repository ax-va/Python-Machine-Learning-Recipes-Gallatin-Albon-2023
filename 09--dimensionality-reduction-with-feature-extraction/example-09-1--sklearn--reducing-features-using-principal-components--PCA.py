"""
Reduce the number of features while retaining the variance (important information) in the data.
->
Use Principal Component Analysis (PCA).

PCA works well if the data is linearly separable.
->
Different classes can be separated by line or (hyper)plane.

See also:
- Scikit-Learn-PCA
https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
"""
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import datasets

# Load the data
digits = datasets.load_digits()
# {'data': array([[ 0.,  0.,  5., ...,  0.,  0.,  0.],
#         [ 0.,  0.,  0., ..., 10.,  0.,  0.],
#         [ 0.,  0.,  0., ..., 16.,  9.,  0.],
#         ...,
#         [ 0.,  0.,  1., ...,  6.,  0.,  0.],
#         [ 0.,  0.,  2., ..., 12.,  0.,  0.],
#         [ 0.,  0., 10., ..., 12.,  1.,  0.]]),
#  'target': array([0, 1, 2, ..., 8, 9, 8]),
#  'frame': None,
#  'feature_names': ['pixel_0_0',
#   'pixel_0_1',
#   'pixel_0_2',
# ...

digits.data.shape
# (1797, 64)

digits.target.shape
# (1797,)

# Standardize the feature matrix
features = StandardScaler().fit_transform(digits.data)
# array([[ 0.        , -0.33501649, -0.04308102, ..., -1.14664746,
#         -0.5056698 , -0.19600752],
#        [ 0.        , -0.33501649, -1.09493684, ...,  0.54856067,
#         -0.5056698 , -0.19600752],
#        [ 0.        , -0.33501649, -1.09493684, ...,  1.56568555,
#          1.6951369 , -0.19600752],
#        ...,
#        [ 0.        , -0.33501649, -0.88456568, ..., -0.12952258,
#         -0.5056698 , -0.19600752],
#        [ 0.        , -0.33501649, -0.67419451, ...,  0.8876023 ,
#         -0.5056698 , -0.19600752],
#        [ 0.        , -0.33501649,  1.00877481, ...,  0.8876023 ,
#         -0.26113572, -0.19600752]])

# Create a PCA that will retain 99% of variance
pca = PCA(
    # n_components > 1 -> Leave the number of components
    # 0 < n_components <= 1 -> Set variance. typically 0.95 or 0.99
    n_components=0.99,
    # Transform the values of each principal component to have zero mean and unit variance
    whiten=True,
    ## Implement a stochastic algorithm to find the first
    ## principal components in often significantly less time
    # svd_solver="randomized",
)

# Conduct PCA
features_pca = pca.fit_transform(features)

# Print results
print("Original number of features:", features.shape[1])
# Original number of features: 64
print("Reduced number of features:", features_pca.shape[1])
# Reduced number of features: 54
