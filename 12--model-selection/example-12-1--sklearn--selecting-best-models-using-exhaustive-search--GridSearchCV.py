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

# Get all CV results
grid_search.cv_results_
# {'mean_fit_time': array([0.01006713, 0.00054245, 0.005725  , 0.00054154, 0.00627165,
#         0.0005527 , 0.0068203 , 0.00057821, 0.0077774 , 0.00059123,
#         0.00894246, 0.00062995, 0.00926266, 0.0006608 , 0.00952744,
#         0.00063963, 0.01009564, 0.00063496, 0.00989614, 0.00068097]),
#  'std_fit_time': array([3.53695458e-03, 2.71275505e-05, 1.15251601e-03, 2.43082211e-05,
#         1.26098048e-03, 5.03147941e-06, 1.57099399e-03, 1.50988060e-05,
#         1.84661691e-03, 2.20912022e-05, 2.12158206e-03, 2.22894723e-05,
#         2.03037409e-03, 3.80654059e-05, 2.21408666e-03, 1.91753765e-05,
#         2.52129206e-03, 1.47337281e-05, 2.34438257e-03, 1.78463274e-05]),
#  'mean_score_time': array([0.00070767, 0.00026054, 0.00029006, 0.00024962, 0.00026412,
#         0.00024047, 0.00027628, 0.00024028, 0.00026512, 0.00024548,
#         0.00031385, 0.00025344, 0.00030746, 0.00027657, 0.0002707 ,
#         0.00024829, 0.00025949, 0.0002461 , 0.0002821 , 0.00025606]),
#  'std_score_time': array([3.31350755e-04, 8.84195479e-06, 1.46991193e-05, 1.55159031e-05,
#         1.29559385e-05, 2.20894524e-06, 1.65369711e-05, 2.19241408e-06,
#         2.01674498e-05, 6.69374757e-06, 5.95799586e-05, 3.66885835e-06,
#         6.32203357e-05, 1.97090128e-05, 2.91949756e-05, 1.17833785e-05,
#         1.65432947e-05, 6.70698203e-06, 1.10131836e-05, 2.96637173e-06]),
#  'param_C': masked_array(data=[1.0, 1.0, 2.7825594022071245, 2.7825594022071245,
#                     7.742636826811269, 7.742636826811269,
#                     21.544346900318832, 21.544346900318832,
#                     59.94842503189409, 59.94842503189409,
#                     166.81005372000593, 166.81005372000593,
#                     464.15888336127773, 464.15888336127773,
#                     1291.5496650148827, 1291.5496650148827,
#                     3593.813663804626, 3593.813663804626, 10000.0, 10000.0],
#               mask=[False, False, False, False, False, False, False, False,
#                     False, False, False, False, False, False, False, False,
#                     False, False, False, False],
#         fill_value='?',
#              dtype=object),
#  'param_penalty': masked_array(data=['l1', 'l2', 'l1', 'l2', 'l1', 'l2', 'l1', 'l2', 'l1',
#                     'l2', 'l1', 'l2', 'l1', 'l2', 'l1', 'l2', 'l1', 'l2',
#                     'l1', 'l2'],
#               mask=[False, False, False, False, False, False, False, False,
#                     False, False, False, False, False, False, False, False,
#                     False, False, False, False],
#         fill_value='?',
#              dtype=object),
#  'params': [{'C': 1.0, 'penalty': 'l1'},
#   {'C': 1.0, 'penalty': 'l2'},
#   {'C': 2.7825594022071245, 'penalty': 'l1'},
#   {'C': 2.7825594022071245, 'penalty': 'l2'},
#   {'C': 7.742636826811269, 'penalty': 'l1'},
#   {'C': 7.742636826811269, 'penalty': 'l2'},
#   {'C': 21.544346900318832, 'penalty': 'l1'},
#   {'C': 21.544346900318832, 'penalty': 'l2'},
#   {'C': 59.94842503189409, 'penalty': 'l1'},
#   {'C': 59.94842503189409, 'penalty': 'l2'},
#   {'C': 166.81005372000593, 'penalty': 'l1'},
#   {'C': 166.81005372000593, 'penalty': 'l2'},
#   {'C': 464.15888336127773, 'penalty': 'l1'},
#   {'C': 464.15888336127773, 'penalty': 'l2'},
#   {'C': 1291.5496650148827, 'penalty': 'l1'},
#   {'C': 1291.5496650148827, 'penalty': 'l2'},
#   {'C': 3593.813663804626, 'penalty': 'l1'},
#   {'C': 3593.813663804626, 'penalty': 'l2'},
#   {'C': 10000.0, 'penalty': 'l1'},
#   {'C': 10000.0, 'penalty': 'l2'}],
#  'split0_test_score': array([1.        , 1.        , 1.        , 1.        , 1.        ,
#         1.        , 1.        , 1.        , 1.        , 1.        ,
#         0.96666667, 1.        , 0.96666667, 1.        , 0.96666667,
#         1.        , 0.96666667, 1.        , 0.96666667, 0.96666667]),
#  'split1_test_score': array([0.96666667, 0.96666667, 1.        , 0.96666667, 1.        ,
#         1.        , 0.96666667, 1.        , 0.96666667, 1.        ,
#         0.96666667, 1.        , 0.96666667, 1.        , 0.96666667,
#         0.96666667, 0.96666667, 0.96666667, 0.96666667, 0.96666667]),
#  'split2_test_score': array([0.93333333, 0.93333333, 0.93333333, 0.93333333, 0.96666667,
#         0.93333333, 0.93333333, 0.93333333, 0.93333333, 0.93333333,
#         0.93333333, 0.96666667, 0.93333333, 0.93333333, 0.93333333,
#         0.93333333, 0.93333333, 0.93333333, 0.93333333, 0.93333333]),
#  'split3_test_score': array([0.9       , 0.9       , 0.9       , 0.9       , 0.93333333,
#         0.9       , 0.93333333, 0.9       , 0.93333333, 0.93333333,
#         0.93333333, 0.93333333, 0.93333333, 0.93333333, 0.93333333,
#         0.93333333, 0.93333333, 0.93333333, 0.93333333, 0.93333333]),
#  'split4_test_score': array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
#         1., 1., 1.]),
#  'mean_test_score': array([0.96      , 0.96      , 0.96666667, 0.96      , 0.98      ,
#         0.96666667, 0.96666667, 0.96666667, 0.96666667, 0.97333333,
#         0.96      , 0.98      , 0.96      , 0.97333333, 0.96      ,
#         0.96666667, 0.96      , 0.96666667, 0.96      , 0.96      ]),
#  'std_test_score': array([0.03887301, 0.03887301, 0.0421637 , 0.03887301, 0.02666667,
#         0.0421637 , 0.02981424, 0.0421637 , 0.02981424, 0.03265986,
#         0.02494438, 0.02666667, 0.02494438, 0.03265986, 0.02494438,
#         0.02981424, 0.02494438, 0.02981424, 0.02494438, 0.02494438]),
#  'rank_test_score': array([12, 12,  5, 12,  1,  5,  5,  5,  5,  3, 15,  1, 15,  3, 15,  5, 15,
#          5, 15, 15], dtype=int32)}
