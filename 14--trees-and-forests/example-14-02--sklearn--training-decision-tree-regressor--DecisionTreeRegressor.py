"""
Train a regression model using a decision tree.
->
Use Scikit-Learn's DecisionTreeRegressor.

By default, potential splits are measured on how much they reduce mean squared error (MSE):
$$
MSE = 1 / n \sum_i^n ({y_true}_i - {y_predicted]_i)
$$

See also:
- Scikit-Learn: Decision Tree Regression
https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html
"""
from sklearn.tree import DecisionTreeRegressor
from sklearn import datasets

# Load data with only two features
diabetes = datasets.load_diabetes()
features = diabetes.data
target = diabetes.target

diabetes.feature_names
# ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
features.shape
# (442, 10)
target.shape
# (442,)
features
# array([[ 0.03807591,  0.05068012,  0.06169621, ..., -0.00259226,
#          0.01990749, -0.01764613],
#        [-0.00188202, -0.04464164, -0.05147406, ..., -0.03949338,
#         -0.06833155, -0.09220405],
#        [ 0.08529891,  0.05068012,  0.04445121, ..., -0.00259226,
#          0.00286131, -0.02593034],
#        ...,
#        [ 0.04170844,  0.05068012, -0.01590626, ..., -0.01107952,
#         -0.04688253,  0.01549073],
#        [-0.04547248, -0.04464164,  0.03906215, ...,  0.02655962,
#          0.04452873, -0.02593034],
#        [-0.04547248, -0.04464164, -0.0730303 , ..., -0.03949338,
#         -0.00422151,  0.00306441]])
target
# array([151.,  75., 141., 206., 135.,  97., 138.,  63., 110., 310., 101.,
#         69., 179., 185., 118., 171., 166., 144.,  97., 168.,  68.,  49.,
# ...
#         84.,  42., 146., 212., 233.,  91., 111., 152., 120.,  67., 310.,
#         94., 183.,  66., 173.,  72.,  49.,  64.,  48., 178., 104., 132.,
#        220.,  57.])

# Create decision tree regressor object
decision_tree = DecisionTreeRegressor(random_state=0)
# Train model
model = decision_tree.fit(features, target)
# Predict the target value for an observation
model.predict([features[0]])
# array([0.])

# Set mean absolute error (MAE)
decision_tree_mae = DecisionTreeRegressor(criterion="absolute_error", random_state=0)
# Train model
model_mae = decision_tree_mae.fit(features, target)
# Predict target value
model_mae.predict([features[0]])
# array([0.])
