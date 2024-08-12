"""
Group observations into clusters of high density.
->
Use DBSCAN clustering.

Idea:
1.  Choose a random observation, $x_i$.
2.  If $x_i$ has a minimum number of close neighbors, it is a part of a cluster.
3.  Repeat recursively step 2 for all $x_i$'s neighbors, then neighbors' neighbors, and so on.
    Thus, get the cluster's core observations.
4.  Once step 3 runs out of nearby observations, choose a new random point and go to step 1.

Finally, any observation close to a cluster but not a core sample is considered part of a cluster.
Also, any observation not close to the cluster is labeled an outlier.
Outlier observations are labeled as -1.

DBSCAN parameters in Scikit-Learn:

- eps:
The maximum distance from an observation for another observation to be considered its neighbor.

- min_samples:
The minimum number of observations less than eps distance
from an observation for it to be considered a core observation.

- metric:
The distance metric used by "eps": "minkowski" with "p" or "euclidean".
"""
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# Load data
iris = datasets.load_iris()
features = iris.data

# Standardize features
features_std = StandardScaler().fit_transform(features)

# Create DBSCAN object
cluster = DBSCAN(n_jobs=-1)
# Train model
model = cluster.fit(features_std)

# cluster membership
model.labels_
# array([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1,  0,
#         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1,
#         0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  1,
#         1,  1,  1,  1,  1, -1, -1,  1, -1, -1,  1, -1,  1,  1,  1,  1,  1,
#        -1,  1,  1,  1, -1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
#        -1,  1, -1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  1, -1,  1, -1,  1,
#         1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1,  1, -1,  1,  1, -1, -1,
#        -1,  1,  1, -1,  1,  1, -1,  1,  1,  1, -1, -1, -1,  1,  1,  1, -1,
#        -1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1,  1])
