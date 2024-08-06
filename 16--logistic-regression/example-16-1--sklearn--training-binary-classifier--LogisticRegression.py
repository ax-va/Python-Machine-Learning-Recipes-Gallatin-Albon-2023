"""
Train a binary (only two classes) classifier model.
->
Use a logistic regression Scikit-Learn's LogisticRegression.

Logistic regression is a widely used binary classifier.
using the composition of a linear model
$$ \beta_0 + \beta_1 x $$
and the sigmoid (also logistic) function
$$ 1 / (1 + \exp{-z}) $$,
such that the conditional probability is equal
$$
P(y_i = 1 | X) = 1 / (1 + \exp{-(\beta_0 + \beta_1 x)})
$$.
"""
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# Load data with only two classes
iris = datasets.load_iris()
features = iris.data[:100, :]
target = iris.target[:100]

# Standardize features
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# Create logistic regression
logistic_regression = LogisticRegression(random_state=0)

# Train model
model = logistic_regression.fit(features_standardized, target)

# Create new observation
new_observation = [[.5, .5, .5, .5]]

# Get predicted probabilities
model.predict_proba(new_observation)
# array([[0.17740549, 0.82259451]])

# P(y_new_observation = 0 | X) ~ 0.177
# P(y_new_observation = 1 | X) ~ 0.823
