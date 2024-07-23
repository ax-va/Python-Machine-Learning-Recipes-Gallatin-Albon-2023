"""
Evaluate an unsupervised learning algorithm for clustering data.
->
Use *silhouette coefficients* to measure the quality of the clusters
(note that this does not measure predictive performance).

Assume that "good" clusters should have very small distances between observations in the same
cluster (i.e., dense clusters) and large distances between the different clusters (i.e.,
well-separated clusters).
->
silhouette coefficients of observation i:
s_i = (b_i -a_i) / max(a_i, b_i),
where
a_i is the mean distance between observation i and all observations of the same class,
b_i is the mean distance between observation i and all observations
from the closest cluster of a different class.
Silhouette coefficient = the mean of all s_i.
->
Silhouette coefficients range between –1 and 1,
with 1 indicating dense, well-separated clusters.

See also:
- Sckit-Learn: silhouette_score
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
"""
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate features matrix
features, _ = make_blobs(
    n_samples=1000,
    n_features=10,
    centers=2,
    cluster_std=0.5,
    shuffle=True,
    random_state=1,
)

# Cluster data using k-means to predict classes
model_kmean = KMeans(n_clusters=2, random_state=1).fit(features)

# Get predicted classes
target_predicted = model_kmean.labels_
# array([1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
# ...
#        1, 1, 0, 0, 0, 0, 1, 1, 1, 0], dtype=int32)

# Evaluate model
silhouette_score(features, target_predicted)
# 0.8916265564072141
