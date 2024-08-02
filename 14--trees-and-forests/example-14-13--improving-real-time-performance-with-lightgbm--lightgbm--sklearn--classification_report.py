"""
Train a gradient boosted tree-based model that is computationally optimized.
->
Use the gradient boosted machine library lightgbm
that is highly optimized for training time, inference, and GPU support.

See also:
- LightGBM documentation
https://lightgbm.readthedocs.io/en/latest/

- Alternative: CatBoost
CatBoost is a machine learning algorithm that uses gradient boosting on decision trees.
It is available as an open source library.
https://catboost.ai/en/docs/
"""
import lightgbm as lgb
import numpy as np
from sklearn import datasets
from sklearn.metrics import classification_report

# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Create dataset
lgb_train = lgb.Dataset(features, target)

# Define parameters
params = {
    'objective': 'multiclass',
    'num_class': 3,
    'verbose': -1,
}

# Train model
model = lgb.train(params, lgb_train)

# Get predictions
observations = np.array(
    [[5, 4, 3, 2],
     [2, 3, 4, 5],
     [2, 4, 3, 5],
     [1, 1, 1, 1],
     [2, 2, 2, 2],
     [3, 3, 3, 3]]
)
model.predict(observations)
# array([[3.12433704e-01, 2.07255306e-01, 4.80310990e-01],
#        [1.02124596e-04, 4.47685754e-03, 9.95421018e-01],
#        [3.24013572e-01, 1.48387246e-01, 5.27599182e-01],
#        [9.49198548e-01, 4.96475113e-02, 1.15394034e-03],
#        [3.46233904e-03, 1.40579710e-03, 9.95131864e-01],
#        [1.36878359e-02, 3.80600780e-03, 9.82506156e-01]])
predictions = np.argmax(model.predict(observations), axis=1)
# array([2, 2, 2, 0, 2, 2])

# Get a classification report
print(classification_report(np.array([2, 2, 2, 0, 2, 2]), predictions))
#               precision    recall  f1-score   support
#
#            0       1.00      1.00      1.00         1
#            2       1.00      1.00      1.00         5
#
#     accuracy                           1.00         6
#    macro avg       1.00      1.00      1.00         6
# weighted avg       1.00      1.00      1.00         6
