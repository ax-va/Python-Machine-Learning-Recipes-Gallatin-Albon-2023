"""
Select the best value for k in a k-nearest neighbors classifier.
->
Use model selection techniques like GridSearchCV with preprocessing using Pipeline.

The bias-variance trade-off:
- k = n -> high bias but low variance (very low model complexity) -> underfitting
- k = 1 -> low bias but high variance (very high model complexity) -> overfitting

See also:
- K Nearest Neighbor and the Bias-variance Trade-off
https://teazrq.github.io/stat542/rlab/knn.html
"""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Create standardizer
standardizer = StandardScaler()

# Create a KNN classifier
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)

# Create a pipeline for proper build-in preprocessing
pipe = Pipeline([("standardizer", standardizer), ("knn", knn)])

# candidate values for k
candidate_values = np.arange(1, 11)
# array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])

# Create space of candidate values for k
search_space = [
    {"knn__n_neighbors": candidate_values},
]

# Use GridSearchCV to conduct five-fold cross-validation
# on KNN classifiers with different values of k given in "search_space".
classifier = GridSearchCV(pipe, search_space, cv=5, verbose=0).fit(features, target)

# the best found classifier
classifier.best_estimator_
# Pipeline(steps=[('standardizer', StandardScaler()),
#                 ('knn', KNeighborsClassifier(n_jobs=-1, n_neighbors=6))])

# the best number of neighbors
classifier.best_estimator_.get_params()["knn__n_neighbors"]
# 6
