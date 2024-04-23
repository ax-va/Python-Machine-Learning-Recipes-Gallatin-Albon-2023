"""
Use k-means clustering (unsupervised learning algorithms) to group similar observations
and output a new feature containing each observation's group membership.
"""
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Make simulated feature matrix
features, target = make_blobs(
    n_samples=50,
    n_features=2,  # number of features: x_1 nad x_2
    centers=3,  # number of clusters
    random_state=1,
)

# Create DataFrame
df = pd.DataFrame(features, columns=["feature_1", "feature_2"])
df.head()
#    feature_1  feature_2
# 0  -9.877554  -3.336145
# 1  -7.287210  -8.353986
# 2  -6.943061  -7.023744
# 3  -7.440167  -8.791959
# 4  -6.641388  -8.075888

# Make k-means clusterer
clusterer = KMeans(n_clusters=3, random_state=0)

# Fit clusterer
clusterer.fit(features)

# Predict values
df["cluster"] = clusterer.predict(features)
df.head()
#    feature_1  feature_2  cluster
# 0  -9.877554  -3.336145        2
# 1  -7.287210  -8.353986        0
# 2  -6.943061  -7.023744        0
# 3  -7.440167  -8.791959        0
# 4  -6.641388  -8.075888        0
