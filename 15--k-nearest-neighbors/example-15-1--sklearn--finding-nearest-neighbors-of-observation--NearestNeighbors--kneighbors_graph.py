"""
Find an observation's k nearest observations (neighbors).
->
Use Scikit-Learn's NearestNeighbors.

Distance metrics:

- Euclidean distance (default for Scikit-Learn's NearestNeighbors)
$$
\sqrt{\sum_i (x_i - y_i)^2}
$$

- Manhattan distance
$$
\sum_i |x_i - y_i|
$$

- Minkowski distance
$$
(\sum_i |x_i - y_i|^p)^{1/p}
$$

where $x$ and $y$ are two observation.

Notice:

When using any learning algorithm based on distance, it is important
to transform features so that they are on the same scale.
Otherwise, for example, if one feature is in millions of dollars and
a second feature is in percentages, the distance calculated will be
biased toward the former.
"""
import numpy as np
from sklearn import datasets
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Load data
iris = datasets.load_iris()
features = iris.data

# When using any learning algorithm based on distance, it is important
# to transform features so that they are on the same scale.
standardizer = StandardScaler()

# Standardize features
features_standardized = standardizer.fit_transform(features)

# To find the two closest observations
nearest_neighbors = NearestNeighbors(n_neighbors=2).fit(features_standardized)

# Find distances and indices of the new-observation's nearest neighbors
distances, indices = nearest_neighbors.kneighbors([[1, 1, 1, 1]])

# shortest distances
distances
# array([[0.49140089, 0.74294782]])

indices
# array([[124, 110]])

# Get the nearest neighbors
features_standardized[indices]
# array([[[1.03800476, 0.55861082, 1.10378283, 1.18556721],
#         [0.79566902, 0.32841405, 0.76275827, 1.05393502]]])

# To find the three closest observations based on Euclidean distance
nearest_neighbors_euclidean = NearestNeighbors(
    n_neighbors=3,
    metric='euclidean',
).fit(features_standardized)

# Use kneighbors_graph to create a matrix indicating
# each observation's nearest neighbors (including itself).
nearest_neighbors_with_self = (
    nearest_neighbors_euclidean
    .kneighbors_graph(features_standardized)
    .toarray()
)

# To print full array without truncation
np.set_printoptions(threshold=np.inf)
# Print three neighbors only for the first observation (including itself)
print(nearest_neighbors_with_self[0, :])
# [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0.]

# Remove 1s marking an observation is a nearest neighbor to itself
for i, x in enumerate(nearest_neighbors_with_self):
    x[i] = 0

# Print two neighbors only for the first observation (excluding itself)
print(nearest_neighbors_with_self[0, :])
# [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0.]
