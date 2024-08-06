"""
Given more than two classes, train a classifier model.
->
Use Scikit-Learn's LogisticRegression with *one-vs-rest* or *multinomial* methods.

- one-vs-rest logistic regression (OvR):
A separate model is trained for each class predicted, whether an observation is that class or not.
It assumes that each classification problem (e.g., class 0 or not) is independent.

- multinomial logistic regression (MLR):
The sigmoid (also logistic) function is replaced with the softmax function:
$$
P(y_i = k | X) = \exp{\beta_k x_i} / \sum_{j=1}^K \exp(\beta_j x_i)
$$,
where $K$ is the total number of classes.
->
The probabilities are more reliable (i.e., better calibrated).
"""
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Standardize features
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# Create one-vs-rest and MLR logistic regressions
logistic_regression_ovr = LogisticRegression(random_state=0, multi_class="ovr")
logistic_regression_mlr = LogisticRegression(random_state=0, multi_class="multinomial")

# Train models
model_ovr = logistic_regression_ovr.fit(features_standardized, target)
model_mlr = logistic_regression_mlr.fit(features_standardized, target)

# Create new observation
new_observation = [[.5, .5, .5, .5]]

# Get predicted probabilities
model_ovr.predict_proba(new_observation)
# array([[0.0387829 , 0.40665354, 0.55456356]])
model_mlr.predict_proba(new_observation)
# array([[0.0198333 , 0.74472208, 0.23544462]])
