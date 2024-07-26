"""
Select the best model by searching over a range of learning algorithms and their respective hyperparameters.
->
Use GridSearchCV with a dictionary of candidate learning algorithms and their hyperparameters.
"""
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Set random seed
np.random.seed(0)
# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

features.shape
# (150, 4)

target.shape
# (150,)

# Create an instance needed as an estimator with a "score" function for GridSearchCV
estimator = Pipeline([("classifier", LogisticRegression())])

# Create dictionary with candidate learning algorithms and their hyperparameters
search_space = [
    {
        "classifier": [LogisticRegression(max_iter=500, solver='liblinear')],
        "classifier__penalty": ['l1', 'l2'],  # classifier__<hyperparameter_name>
        "classifier__C": np.logspace(0, 4, 10)  # classifier__<hyperparameter_name>
    },
    {
        "classifier": [RandomForestClassifier()],
        "classifier__n_estimators": [10, 100, 1000],  # classifier__<hyperparameter_name>
        "classifier__max_features": [1, 2, 3]  # classifier__<hyperparameter_name>
    },
]

# Create grid search
grid_search = GridSearchCV(estimator=estimator, param_grid=search_space, cv=5, verbose=0)

# Fit grid search
best_model = grid_search.fit(features, target)
# best model
best_model.best_estimator_
# Pipeline(steps=[('classifier',
#                  LogisticRegression(C=7.742636826811269, max_iter=500,
#                                     penalty='l1', solver='liblinear'))])

best_model.best_estimator_.get_params()["classifier"]
# LogisticRegression(C=7.742636826811269, max_iter=500, penalty='l1',
#                    solver='liblinear')

# Predict target vector
best_model.predict(features)
# array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#        2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
