"""
Fetch nearest neighbors for big data at low latency.
->
Use an *approximate nearest neighbors* (ANN) based search with Facebook's faiss library:
https://github.com/facebookresearch/faiss
https://github.com/facebookresearch/faiss/blob/main/INSTALL.md

That is imprecise but fast method.

Install using pip:
$ pip install faiss-cpu

IVF:
An inverted file index (IVF) works by using clustering
to limit the scope of the search space for nearest neighbors.
IVF uses Voronoi tessellations to partition the search space
into a number of distinct areas (or clusters), each of which
contains only a small subset of the total observations.

See also:
- Nearest Neighbor Indexes for Similarity Search
https://www.pinecone.io/learn/series/faiss/vector-indexes/
"""
import faiss
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# Load data
iris = datasets.load_iris()
features = iris.data
# Create standardizer
standardizer = StandardScaler()
# Standardize features
features_standardized = standardizer.fit_transform(features)

# Set faiss parameters
n_features = features_standardized.shape[1]  # 4
nlist = 3  # number of clusters
k = 2

# Create an inverted file index (IVF).
# That consists of search scope reduction through clustering.
quantizer = faiss.IndexFlatIP(n_features)
index = faiss.IndexIVFFlat(quantizer, n_features, nlist)

# Train the index and add feature vectors
index.train(features_standardized)
index.add(features_standardized)

# Create an observation
new_observation = np.array([[1, 1, 1, 1]])

# Search the index for the 2 nearest neighbors
distances, indices = index.search(new_observation, k)

# Show the feature vectors for the two nearest neighbors
np.array([list(features_standardized[i]) for i in indices[0]])
# array([[1.03800476, 0.55861082, 1.10378283, 1.18556721],
#        [0.79566902, 0.32841405, 0.76275827, 1.05393502]])
