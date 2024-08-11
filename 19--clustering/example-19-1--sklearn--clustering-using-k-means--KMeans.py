"""
Group observations into k groups.
->
Use k-means clustering.

Steps:
1. Create k cluster "center" points at random locations.
2. For each observation:
    a. Calculate the distance between each observation and the k center points.
    b. Assign the observation to the cluster of the nearest center point.
3. Move the old centers to the means (i.e., new centers) of their respective clusters.
4. Repeat steps 2 and 3 until no observation changes in cluster membership.

Assumptions:
1. K-means clustering assumes the clusters are convex shaped (e.g., a circle, a sphere).
2. All features are equally scaled. (Use "StandardScaler().fit_transform(features)").
3. The groups are balanced (i.e., have roughly the same number of observations).

See also:
- Introduction to K-means Clustering
https://blogs.oracle.com/ai-and-datascience/post/introduction-to-k-means-clustering
"""
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load data
iris = datasets.load_iris()
features = iris.data

# Standardize features
features_std = StandardScaler().fit_transform(features)

# Create k-means object
cluster = KMeans(
    # number of clusters k to set by the user as a hyperparameter;
    # if unknown, use e.g. *silhouette coefficients*
    n_clusters=3,
    random_state=0,
    n_init="auto",
)
# Train model
model = cluster.fit(features_std)

# predicted classes
model.labels_
# array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 2, 2, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2,
#        0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 2, 2, 2, 0, 2, 2, 2,
#        2, 2, 2, 0, 0, 2, 2, 2, 2, 0, 2, 0, 2, 0, 2, 2, 0, 2, 2, 2, 2, 2,
#        2, 0, 0, 2, 2, 2, 0, 2, 2, 2, 0, 2, 2, 2, 0, 2, 2, 0], dtype=int32)

# Rename classes (if necessary) to compare with true classes
indices_0 = model.labels_==1
indices_1 = model.labels_==0
indices_2 = model.labels_==2

model.labels_[indices_0] = 0
model.labels_[indices_1] = 1
model.labels_[indices_2] = 2

# renamed classes
model.labels_
# array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 2, 2, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2,
#        1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2,
#        2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2,
#        2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 1], dtype=int32)

# true classes
iris.target
# array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

# Predict observation's cluster
model.predict([[0.8, 0.8, 0.8, 0.8]])
# array([2], dtype=int32)

# cluster centers for non-renamed classes, i.e. initially ordered as 1, 0, 2
model.cluster_centers_
# array([[-0.05021989, -0.88337647,  0.34773781,  0.2815273 ],
#        [-1.01457897,  0.85326268, -1.30498732, -1.25489349],
#        [ 1.13597027,  0.08842168,  0.99615451,  1.01752612]])
