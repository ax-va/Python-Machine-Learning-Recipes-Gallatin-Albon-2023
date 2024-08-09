"""
Identify which observations are the support vectors of the decision hyperplane.
->
Train the model, then use "support_vectors_" to get the support vectors,
"support_" to get their indices, and "n_support_" to get the number
of support vectors belonging to each class.

Support vector machines get their name from the fact that the hyperplane is being
determined by a relatively small number of observations, called the *support vectors*.
"""
import numpy as np
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# Load data with only two classes
iris = datasets.load_iris()
features = iris.data[:100, :]
target = iris.target[:100]
np.unique(target)
# array([0, 1])

# Standardize features
features_standardized = StandardScaler().fit_transform(features)

# Create support vector classifier object
svc = SVC(kernel="linear", random_state=0)
# Train classifier
model = svc.fit(features_standardized, target)

# support vectors
model.support_vectors_
# array([[-0.5810659 ,  0.42196824, -0.80497402, -0.50860702],
#        [-1.52079513, -1.67737625, -1.08231219, -0.86427627],
#        [-0.89430898, -1.4674418 ,  0.30437864,  0.38056609],
#        [-0.5810659 , -1.25750735,  0.09637501,  0.55840072]])

# indices of the support vectors
model.support_
#  array([23, 41, 57, 98], dtype=int32)

# number of support vectors belonging to each class
model.n_support_
# array([2, 2], dtype=int32)
