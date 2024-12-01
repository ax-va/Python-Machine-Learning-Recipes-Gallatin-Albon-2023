"""
Train a support vector machine classifier in the presence of imbalanced classes.
->
Increase the penalty for misclassifying the smaller class setting 'class_weight="balanced"'.

The general idea is to increase the penalty $C$ for misclassifying minority classes
to prevent them from being "overwhelmed" by the majority classes by weighting penalties:
$$
C_j = C w_j
$$,
where $w_j = n / (k n_j)$,
$n$ is the number of observations,
$n_j$ is the number of observations of class $j$,
$k$ is the total number of classes, and
$C_j$ is the penalty for misclassifying for class $j$.

In Scikit-Learn, it is done by setting 'class_weight="balanced"'.
"""
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load data with only two classes
iris = datasets.load_iris()
features = iris.data[:100, :]
target = iris.target[:100]
np.unique(target)
# array([0, 1])

# Make class highly imbalanced by removing first 40 observations
features_imbalanced = features[40:, :]
target_imbalanced = target[40:]

# Standardize features
features_standardized = StandardScaler().fit_transform(features)

# Create support vector classifier
svc = SVC(kernel="linear", class_weight="balanced", C=1.0, random_state=0)
# Train classifier
model = svc.fit(features_standardized, target)
