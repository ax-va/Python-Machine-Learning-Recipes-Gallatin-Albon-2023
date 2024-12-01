"""
Group observations using a hierarchy of clusters.
->
Use agglomerative clustering.

Idea:
In agglomerative clustering, all observations start as their own clusters.
Next, clusters meeting some criteria are merged.
This process is repeated, growing clusters until some end point is reached.

AgglomerativeClustering parameters:

- linkage:
Determine the merging strategy to minimize:
-- variance of merged clusters ("ward");
-- average distance between observations from pairs of clusters ("average");
-- maximum distance between observations from pairs of clusters ("complete").

- affinity:
Determine the distance metric used for "linkage" ("minkowski", "euclidean", etc.).

- n_clusters:
Determine a final number of clusters.
"""
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

# Load data
iris = datasets.load_iris()
features = iris.data

# Standardize features
features_std = StandardScaler().fit_transform(features)

# Create agglomerative clustering
cluster = AgglomerativeClustering(n_clusters=3)

# Train model
model = cluster.fit(features_std)

# cluster membership
model.labels_
# array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1,
#        1, 1, 1, 1, 1, 1, 0, 0, 0, 2, 0, 2, 0, 2, 0, 2, 2, 0, 2, 0, 2, 0,
#        2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 2, 0, 0, 2,
#        2, 2, 2, 0, 2, 2, 2, 2, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
