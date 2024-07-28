"""
Create a simple baseline regression model to comprise with other models.
->
Use DummyRegressor to create a simple model to use as a baseline.

R-squared score:
$$
R^2 = 1 - \sum_i ({y_true_value}_i - {y_predicted_value}_i) / ({y_true_value}_i - {y_mean})
$$

The closer $R^2$ is to 1, the more of the variance
in the predicted target vector that is explained by the features.
"""
from sklearn.datasets import load_wine
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load data
wine = load_wine()
# Create features
features, target = wine.data, wine.target

# Make test and training split
features_train, features_test, target_train, target_test = train_test_split(
features, target, random_state=0
)
features_train.shape[0] / features.shape[0] * 100
# 74.71910112359551
features_test.shape[0] / features.shape[0] * 100
# 25.280898876404496

# Create a dummy regressor
dummy = DummyRegressor(strategy='mean')
# "Train" dummy regressor
dummy.fit(features_train, target_train)
# Get R-squared score
dummy.score(features_test, target_test)
# -0.0480213580840978

# Create dummy regressor that predicts 1s for everything
dummy_always_ones = DummyRegressor(strategy='constant', constant=1)
# "Train" dummy regressor
dummy_always_ones.fit(features_train, target_train)
# Get R-squared score
dummy_always_ones.score(features_test, target_test)
# -0.06299212598425186

# Compare the dummy regressors with another model
linreg = LinearRegression()
# Train linear regression model
linreg.fit(features_train, target_train)
# Get R-squared score
linreg.score(features_test, target_test)
# 0.804353263176954
