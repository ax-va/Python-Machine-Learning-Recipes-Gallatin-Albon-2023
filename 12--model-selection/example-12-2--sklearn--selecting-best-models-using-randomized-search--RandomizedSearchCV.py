"""
Select the best model with a computationally cheaper method than exhaustive search.
->
Use Scikit-Learn's RandomizedSearchCV to search over a specific number of random combinations
of hyperparameter values from user-supplied distributions (e.g., normal, uniform) and specified lists of values.

See also:
- Scikit Learn: RandomizedSearchCV
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html

- Random Search for Hyper-Parameter Optimization
https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf
"""
from scipy.stats import uniform
from sklearn import linear_model, datasets
from sklearn.model_selection import RandomizedSearchCV

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

# Create range of candidate regularization penalty hyperparameter values
penalty = ['l1', 'l2']

# Create distribution of candidate regularization hyperparameter values
C = uniform(loc=0, scale=4)
# <scipy.stats._distn_infrastructure.rv_continuous_frozen at 0x71bccb04fb50>

# Define a uniform distribution between 0 and 4, sample 10 values
uniform(loc=0, scale=4).rvs(10)
# array([3.99726173, 1.85772094, 0.24802696, 2.65398283, 0.16118695,
#        2.01494821, 2.80447024, 2.45042605, 2.71926183, 0.52943067])

# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty)
# {'C': <scipy.stats._distn_infrastructure.rv_continuous_frozen at 0x71bccb205cd0>,
#  'penalty': ['l1', 'l2']}

# Create randomized search
randomized_search = RandomizedSearchCV(
    logreg, hyperparameters,
    random_state=1,
    n_iter=100,  # number of sampled combinations of hyperparameters
    cv=5,
    verbose=0,
    n_jobs=-1,
)

# Fit randomized search
best_model = randomized_search.fit(features, target)
# best model
best_model.best_estimator_
# LogisticRegression(C=1.668088018810296, max_iter=500, penalty='l1',
#                    solver='liblinear')

# View best hyperparameters
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
# Best Penalty: l1
print('Best C:', best_model.best_estimator_.get_params()['C'])
# Best C: 1.668088018810296

# Predict target vector
best_model.predict(features)
# array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2,
#        2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
