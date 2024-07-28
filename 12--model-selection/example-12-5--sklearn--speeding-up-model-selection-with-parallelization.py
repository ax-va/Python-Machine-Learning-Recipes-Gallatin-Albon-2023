"""
Speed up model selection.
->
Use all the machine cores by setting n_jobs=-1 to train multiple models simultaneously.

Scikit-Learn can simultaneously train models up to the number of cores on the machine.
The parameter n_jobs defines the number of models to train in parallel.

See an experiment below.
"""
import numpy as np
from sklearn import linear_model, datasets
from sklearn.model_selection import GridSearchCV

# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Create logistic regression
logreg = linear_model.LogisticRegression(max_iter=500, solver='liblinear')
# Create range of candidate regularization penalty hyperparameter values
penalty = ["l1", "l2"]
# Create range of candidate values for C
C = np.logspace(0, 4, 1000)
# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty)

# Create grid search
grid_search = GridSearchCV(
    logreg,
    hyperparameters,
    cv=5,
    n_jobs=-1,
    verbose=1,
)

# Fit grid search
best_model = grid_search.fit(features, target)
# Fitting 5 folds for each of 2000 candidates, totalling 10000 fits

# best model
best_model.best_estimator_
# LogisticRegression(C=5.926151812475554, max_iter=500, penalty='l1',
#                    solver='liblinear')

# # experiment: using four cores in a machine speeds up execution time by four times

# Check the execution time in IPython:
# n_jobs=-1 to use all the cores available
"""
%timeit GridSearchCV(logreg, hyperparameters, cv=5, n_jobs=-1, verbose=1).fit(features, target)
Fitting 5 folds for each of 2000 candidates, totalling 10000 fits
Fitting 5 folds for each of 2000 candidates, totalling 10000 fits
Fitting 5 folds for each of 2000 candidates, totalling 10000 fits
Fitting 5 folds for each of 2000 candidates, totalling 10000 fits
Fitting 5 folds for each of 2000 candidates, totalling 10000 fits
Fitting 5 folds for each of 2000 candidates, totalling 10000 fits
Fitting 5 folds for each of 2000 candidates, totalling 10000 fits
Fitting 5 folds for each of 2000 candidates, totalling 10000 fits
9.58 s ± 896 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
"""

# By default, n_jobs=1
"""
%timeit GridSearchCV(logreg, hyperparameters, cv=5, n_jobs=1, verbose=1).fit(features, target)
Fitting 5 folds for each of 2000 candidates, totalling 10000 fits
Fitting 5 folds for each of 2000 candidates, totalling 10000 fits
Fitting 5 folds for each of 2000 candidates, totalling 10000 fits
Fitting 5 folds for each of 2000 candidates, totalling 10000 fits
Fitting 5 folds for each of 2000 candidates, totalling 10000 fits
Fitting 5 folds for each of 2000 candidates, totalling 10000 fits
Fitting 5 folds for each of 2000 candidates, totalling 10000 fits
Fitting 5 folds for each of 2000 candidates, totalling 10000 fits
47.6 s ± 341 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
"""
