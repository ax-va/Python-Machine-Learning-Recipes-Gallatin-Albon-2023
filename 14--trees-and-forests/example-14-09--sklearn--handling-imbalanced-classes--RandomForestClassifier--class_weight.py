"""
Given a target vector with highly imbalanced classes, train a random forest model.
->
Train a decision tree or random forest model with class_weight="balanced".

The class proportions can also be passed manually to class_weight,
e.g., class_weight={"male": 0.2, "female": 0.8}.

When passing class_weight="balanced", classes are automatically weighted
inversely proportional to how frequently they appear in the data:
$$
w_j = n / (k n_j)
$$,
where
$n$ is the number of observations,
$n_j$ is the number of observations in class $j$,
and $k$ is the total number of classes.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target
# array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
target_unique, target_counts = np.unique(target, return_counts=True)
dict(zip(target_unique, target_counts))
# {0: 50, 1: 50, 2: 50}

# Make class highly imbalanced by removing first 40 observations
features_imbalanced = features[40:, :]
target_imbalanced = target[40:]
# array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
#        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
target_imbalanced_unique, target_imbalanced_counts = np.unique(target_imbalanced, return_counts=True)
dict(zip(target_imbalanced_unique, target_imbalanced_counts))
# {0: 10, 1: 50, 2: 50}

# Create target vector indicating if class 0, otherwise 1
target_more_imbalanced = np.where((target_imbalanced == 0), 0, 1)
# array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
target_more_imbalanced_unique, target_imbalanced_counts = np.unique(target_more_imbalanced, return_counts=True)
dict(zip(target_more_imbalanced_unique, target_imbalanced_counts))
# {0: 10, 1: 100}

# Create random forest classifier object
random_forest = RandomForestClassifier(
    random_state=0,
    n_jobs=-1,
    class_weight="balanced",
)

# Train model
model = random_forest.fit(features_imbalanced, target_more_imbalanced)

# Calculate weight for small class
110/(2*10)
# 5.5
# Calculate weight for large class
110/(2*100)
# 0.55
