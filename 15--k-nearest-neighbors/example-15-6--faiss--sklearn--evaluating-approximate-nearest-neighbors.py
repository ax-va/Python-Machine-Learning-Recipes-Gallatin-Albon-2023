"""
Know how approximate nearest neighbors (ANN) compares to exact k-mean nearest neighbors (KNN).
->
Compute the *recall @k* nearest neighbors of the ANN as compared to the KNN.

*Recall @k* is a common metric defined as the number of items returned by the ANN
at some k nearest neighbors that also appear in the exact nearest neighbors at the same k,
divided by k.

See also:
- Vertex AI Vector Search
https://cloud.google.com/vertex-ai/docs/vector-search/overview?hl=de#why_does_ann_perform_approximate_matches_instead_of_exact_matches
"""
import faiss
import numpy as np
from sklearn import datasets
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# number of nearest neighbors
k = 10

# Load data
iris = datasets.load_iris()
features = iris.data

# Create standardizer
standardizer = StandardScaler()
# Standardize features
features_standardized = standardizer.fit_transform(features)

# Create KNN with 10 NN
knn = NearestNeighbors(n_neighbors=k).fit(features_standardized)

# Set faiss parameters
n_features = features_standardized.shape[1]  # 4
nlist = 3  # number of clusters to create

# Create an IVF index
quantizer = faiss.IndexFlatIP(n_features)
index = faiss.IndexIVFFlat(quantizer, n_features, nlist)

# Train the index and add feature vectors
index.train(features_standardized)
index.add(features_standardized)
index.nprobe = 1  # number of clusters to search

# Create an observation
new_observation = np.array([[ 1, 1, 1, 1]])

# Find distances and indices of the observation's exact nearest neighbors
knn_distances, knn_indices = knn.kneighbors(new_observation)
knn_distances
# array([[0.49140089, 0.74294782, 0.75692864, 0.76371162, 0.84505638,
#         0.85152876, 0.86130084, 0.87959976, 0.91297623, 0.96472321]])
knn_indices
# array([[124, 110, 148, 136, 144, 143, 120, 115,  56, 139]])

# Search the index for the k approximate nearest neighbors
ivf_distances, ivf_indices = index.search(new_observation, k)
ivf_distances
# array([[0.24147487, 0.55197144, 0.572941  , 0.58325547, 0.7141203 ,
#         0.7251011 , 0.74183905, 0.7736957 , 0.83352566, 0.9306909 ]],
#       dtype=float32)
ivf_indices
# array([[124, 110, 148, 136, 144, 143, 120, 115,  56, 139]])

# Get the set overlap
recalled_items = set(list(knn_indices[0])) & set(list(ivf_indices[0]))
# {56, 110, 115, 120, 124, 136, 139, 143, 144, 148}

# Print the recall
print(f"Recall @k={k}: {len(recalled_items) / k * 100}%")
# Recall @k=10: 100.0%
