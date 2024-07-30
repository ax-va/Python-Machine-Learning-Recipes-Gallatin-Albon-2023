"""
Train a classification model using a "forest" of randomized decision trees.
->
Use Scikit-Learn's RandomForestClassifier to train a random forest classification model.

Motivation:

Decision trees tend to fit the training data too closely what leads to overfitting.
In a *random forest*, many decision trees are trained, but each tree receives only
a *bootstrapped sample* of observations (a random sample of observations with replacement),
and each node considers only a subset of features when determining the best split.
This forest of randomized decision trees votes to determine the predicted class.

Additional parameters:

- max_features:
determines the maximum number of features to be considered at each node and takes a number
of arguments including integers (number of features), floats (percentage of features),
and "sqrt" (square root of the number of features).

By default, max_features is set to "auto", which acts the same as "sqrt".

- bootstrap:
sets whether the subset of observations considered for a tree is created
using sampling with replacement (the default setting) or without replacement.

- n_estimators:
sets the number of decision trees to include in the forest.

See also:
- Random Forests
https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target
iris.target_names
# array(['setosa', 'versicolor', 'virginica'], dtype='<U10')

# Create random forest classifier object
random_forest = RandomForestClassifier(
    random_state=0,
    n_jobs=-1,  # Use all available CPU cores to train trees parallely
)

# Train model
model = random_forest.fit(features, target)

# Predict observation's class
model.predict([[5, 4, 3, 2]])
# array([1])

# Use entropy as a metric
random_forest_entropy = RandomForestClassifier(
    criterion="entropy", random_state=0,
)

# Train model
model_entropy = random_forest_entropy.fit(features, target)
