"""
Select the best model by searching over a range of hyperparameters.
->
Use Scikit-Learn's GridSearchCV that is a brute-force approach to model selection using k-fold cross-validation.

By default, after identifying the best hyperparameters, GridSearchCV
will re-train a model using the best hyperparameters  on the entire dataset.

One GridSearchCV parameter is verbose that determines the number of messages
outputted during the search, with 0 showing no output, and 1 to 3 outputting additional messages.

See also:
- Scikit-Learn: GridSearchCV
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
"""
import numpy as np
from sklearn import linear_model, datasets
from sklearn.model_selection import GridSearchCV

# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target
features.shape
# (150, 4)
target.shape
# (150,)

# Create logistic regression
logreg = linear_model.LogisticRegression(max_iter=500, solver='liblinear')

# Create range of candidate penalty hyperparameter values
penalty = ['l1', 'l2']

# Create range of candidate regularization hyperparameter values
C = np.logspace(0, 4, 10)
# array([1.00000000e+00, 2.78255940e+00, 7.74263683e+00, 2.15443469e+01,
#        5.99484250e+01, 1.66810054e+02, 4.64158883e+02, 1.29154967e+03,
#        3.59381366e+03, 1.00000000e+04])

# Create dictionary of hyperparameter candidates
hyperparameters = dict(C=C, penalty=penalty)
# {'C': array([1.00000000e+00, 2.78255940e+00, 7.74263683e+00, 2.15443469e+01,
#         5.99484250e+01, 1.66810054e+02, 4.64158883e+02, 1.29154967e+03,
#         3.59381366e+03, 1.00000000e+04]),
#  'penalty': ['l1', 'l2']}

# Create grid search
grid_search = GridSearchCV(logreg, hyperparameters, cv=5, verbose=0)
# Fit grid search with 10 values of C, 2 values of regularization penalty,
# and 5 folds, i.e. with 10*2*5=100 hyperparameters.
best_model = grid_search.fit(features, target)
# By default, after identifying the best hyperparameters, GridSearchCV
# will re-train a model using the best hyperparameters  on the entire dataset.

# the best model
best_model.best_estimator_
# LogisticRegression(C=7.742636826811269, max_iter=500, penalty='l1',
#                    solver='liblinear')

# View best hyperparameters
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
# Best Penalty: l1
print('Best C:', best_model.best_estimator_.get_params()['C'])
# Best C: 7.742636826811269

# Predict target vector
best_model.predict(features)
# array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#        2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
