"""
Train a classifier model on a very large set of data.
->
Train a logistic regression using Scikit-Learn's LogisticRegression
with the *stochastic average gradient* (SAG) solver.
"""
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

# SAG it is very sensitive to feature scaling,
# so standardizing the features is particularly important.
# Standardize features.
features_standardized = StandardScaler().fit_transform(features)

# Create logistic regression object
logistic_regression = LogisticRegression(random_state=0, solver="sag")

# Train model
model = logistic_regression.fit(features_standardized, target)