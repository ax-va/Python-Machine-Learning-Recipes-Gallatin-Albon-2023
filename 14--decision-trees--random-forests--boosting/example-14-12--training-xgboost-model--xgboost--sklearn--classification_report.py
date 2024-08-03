"""
Train a tree-based model with high predictive power.
->
Use Extreme Gradient Boosting (XGBoost) from the xgboost Python library
that is a reliable algorithm for improving performance
beyond that of typical random forests or gradient boosted machines.

See also:
- XGBoost Documentation
https://xgboost.readthedocs.io/en/stable/
"""
import numpy as np
import xgboost as xgb
from sklearn import datasets
from sklearn.metrics import classification_report

# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target
# array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

features.shape
# (150, 4)
target.shape
# (150,)

# Create dataset
xgb_train = xgb.DMatrix(features, label=target)
# <xgboost.core.DMatrix at 0x7f927a432290>

# Define parameters
param = {
    'objective': 'multi:softprob',
    'num_class': 3
}

# Train model
model = xgb.train(param, xgb_train)

# Get predictions
observations = xgb.DMatrix(
    [[5, 4, 3, 2],
     [2, 3, 4, 5],
     [2, 4, 3, 5],
     [1, 1, 1, 1],
     [2, 2, 2, 2],
     [3, 3, 3, 3]]
)
model.predict(observations)
# array([[0.05505896, 0.43452   , 0.51042104],
#        [0.05844222, 0.39977247, 0.5417853 ],
#        [0.05505896, 0.43452   , 0.51042104],
#        [0.95302665, 0.02447841, 0.02249493],
#        [0.80587876, 0.02069893, 0.17342232],
#        [0.05844222, 0.39977247, 0.5417853 ]], dtype=float32)
predictions = np.argmax(model.predict(observations), axis=1)
# array([2, 2, 2, 0, 0, 2])

# Get a classification report
print(classification_report(np.array([2, 2, 2, 0, 0, 2]), predictions))
#               precision    recall  f1-score   support
#
#            0       1.00      1.00      1.00         2
#            2       1.00      1.00      1.00         4
#
#     accuracy                           1.00         6
#    macro avg       1.00      1.00      1.00         6
# weighted avg       1.00      1.00      1.00         6
