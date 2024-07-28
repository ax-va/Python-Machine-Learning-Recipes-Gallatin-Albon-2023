"""
Evaluate the performance of a model found through model selection.
->
Use nested cross-validation to avoid biased evaluation.

Problem:
After using the data to select the best hyperparameter values,
the same data cannot be used to evaluate the model's performance.

Solution:
Use nested cross-validation so that the "inner" cross-validation selects the best model,
while the "outer" cross-validation provides an unbiased evaluation of the model's performance.
"""
import numpy as np
from sklearn import linear_model, datasets
from sklearn.model_selection import GridSearchCV, cross_val_score

# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Create logistic regression
logreg = linear_model.LogisticRegression(max_iter=500, solver='liblinear')
# Create range of 20 candidate values for C
C = np.logspace(0, 4, 20)
# Create hyperparameter options
hyperparameters = dict(C=C)

# Create the inner cross-validation
grid_search = GridSearchCV(
    logreg, hyperparameters,
    cv=5,
    n_jobs=-1,
    verbose=0,
)

# Wrap the inner cross-validation in the outer cross-validation
scores = cross_val_score(grid_search, features, target)
# array([1.        , 1.        , 0.93333333, 0.93333333, 1.        ])

# Output the average score
scores.mean()
# 0.9733333333333334
