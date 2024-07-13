"""
Simulate data to be used with clustering techniques.
"""
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate features matrix and target vector
features, target = make_blobs(
    n_samples=100,
    n_features=2,  # number of features: x_1 nad x_2
    centers=3,  # number of clusters
    cluster_std=0.5,
    shuffle=True,
    random_state=1,
)

print('Feature Matrix\n', features[:5])
# Feature Matrix
#  [[ -1.22685609   3.25572052]
#  [ -9.57463218  -4.38310652]
#  [-10.71976941  -4.20558148]
#  [ -9.88266514  -3.57234296]
#  [ -5.80071933  -8.27754549]]
print('Target Vector\n', target[:5])
# Target Vector
#  [0 1 1 1 2]

plt.scatter(features[:, 0], features[:, 1], c=target)
plt.savefig('example-02-02-3--sklearn--simulating-data--make_blobs.svg')
plt.close()
