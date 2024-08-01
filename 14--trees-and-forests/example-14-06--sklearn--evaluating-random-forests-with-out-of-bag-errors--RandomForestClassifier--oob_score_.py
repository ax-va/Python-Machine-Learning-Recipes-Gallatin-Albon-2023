"""
Evaluate a random forest model without using cross-validation.
->
Calculate the model's out-of-bag score.

Motivation:

Each decision tree in a random forest is trained using a bootstrapped subset of observations.
->
For every tree, there is a separate subset of observations not being used to train that tree.
Such a subset is called *out-of-bag (OOB)* observations and
can be used as a test set for a subset of trees not trained using those observations.
->
The overall score (OOB score) is calculated, and thus,
OOB score estimation is an alternative to cross-validation.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Create random forest classifier object
random_forest = RandomForestClassifier(
    random_state=0,
    n_estimators=1000,
    oob_score=True,  # Compute OOB scores
    n_jobs=-1,
)

# Train model
model = random_forest.fit(features, target)

# View out-of-bag-error
random_forest.oob_score_
#  0.9533333333333334
