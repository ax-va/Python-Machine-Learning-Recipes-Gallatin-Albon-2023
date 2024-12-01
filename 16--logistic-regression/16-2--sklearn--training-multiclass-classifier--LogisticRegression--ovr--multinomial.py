"""
Given more than two classes, train a classifier model.
->
Use Scikit-Learn's LogisticRegression with *one-vs-rest* or *multinomial* methods.

- *one-vs-rest* logistic regression (OvR):
A separate model is trained for each class predicted, whether an observation is that class or not.
It assumes that each classification problem (e.g., class 0 or not) is independent.

- *multinomial logistic regression* (MLR):
The sigmoid (also logistic) function is replaced with the softmax function:
$$
P(y_i = k | X) = \exp{\beta_k x_i} / \sum_{j=1}^K \exp(\beta_j x_i)
$$,
where $\beta_j$ and $x_i$ are vectors,
the index $i$ denotes an observation index, and
$K$ is the total number of classes.
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

# logistic regression parameters "intercept_" (3 classes)
model_ovr.intercept_
# array([-2.4782882 , -0.9387282 , -3.80186027])
model_mlr.intercept_
# array([-0.20463897,  2.07455834, -1.86991938])

# logistic regression parameters "coef_" (3 classes and 4 features)
model_ovr.coef_
# array([[-1.05603354,  1.22597119, -1.76482206, -1.62922907],
#        [ 0.13599151, -1.27438121,  0.79705947, -0.91626954],
#        [ 0.14017374, -0.51464011,  2.48068262,  3.14066798]])
model_mlr.coef_
# array([[-1.07659238,  1.15977648, -1.92814138, -1.81227307],
#        [ 0.5899243 , -0.36263595, -0.36461312, -0.82700241],
#        [ 0.48666809, -0.79714052,  2.2927545 ,  2.63927548]])
