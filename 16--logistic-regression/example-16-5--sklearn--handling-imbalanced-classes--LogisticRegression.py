"""
Handle imbalanced classes using Scikit-Learn's LogisticRegression.
->
Passing class_weight="balanced" to Scikit-Learn's LogisticRegression
will weight classes inversely proportional to their frequency:
$$ w_j = n / (n_j k) $$,
where
$w_j$ is the weight to class $j$,
$n$ is the number of observations,
$n_j$ is the number of observations of class $j$, and
$k$ is the total number of classes.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Make class highly imbalanced by removing first 40 observations
features_imbalanced= features[40:, :]
target_imbalanced = target[40:]
features_imbalanced.shape
# (110, 4)
target_imbalanced.shape
# (110,)

# Create target vector indicating if class 0, otherwise 1
target_very_imbalanced = np.where((target_imbalanced == 0), 0, 1)
np.unique(target_very_imbalanced)
# array([0, 1])

# Standardize features
features_imbalanced_standardized = StandardScaler().fit_transform(features_imbalanced)

# Create decision tree regression object
logistic_regression = LogisticRegression(random_state=0, class_weight="balanced")

# Train model
model = logistic_regression.fit(features_imbalanced_standardized, target_very_imbalanced)
