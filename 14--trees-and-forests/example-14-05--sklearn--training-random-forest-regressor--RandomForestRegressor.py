"""
Train a regression model using a "forest" of randomized decision trees.
->
Use Scikit-Learn's RandomForestRegressor.

Additional parameters:

- max_features:
sets the maximum number of features to consider at each node;
defaults to "p" features, where "p" is the total number of features.

- bootstrap:
sets whether to sample with replacement; defaults to True.

- n_estimators:
sets the number of decision trees to construct; defaults to 10.

See also:
- Scikit-Learn: RandomForestRegressor
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
"""
from sklearn.ensemble import RandomForestRegressor
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

# Create random forest regressor
random_forest = RandomForestRegressor(random_state=0, n_jobs=-1)
# Train model
model = random_forest.fit(features, target)
# Predict the target value for an observation
model.predict([features[0]])
# array([175.48])
