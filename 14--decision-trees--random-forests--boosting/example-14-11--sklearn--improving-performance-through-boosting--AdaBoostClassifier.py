"""
Get a model with better performance than decision trees or random forests.
->
Train a boosted model using AdaBoostClassifier or AdaBoostRegressor.

Motivation:
An alternative approach to random forest,
and often more powerful, is called *boosting*.

Steps in AdaBoost:

1.
Choose "weak" models, 50 models as default in Scikit-Learn.

2.
Assign every observation, x_i,
an initial weight value, w_i = 1 / n (priority to train),
where n is the total number of observations in the data.

3.
Train a "weak" model (most often a shallow decision tree,
sometimes called a stump) on the data.

4.
For each observation:
If weak model predicts x_i correctly, w_i is decreased.
If weak model predicts xi incorrectly, wi is increased.

5.
Repeat training new weak models where observations
with greater w_i are given greater priority
until the data is perfectly predicted or
a preset number of weak models has been trained.

Parameters of AdaBoost in Scikit-Learn:

- estimator:
a learning algorithm to use to train the weak models, a decision tree as default.

- n_estimators:
the number of models to iteratively train.

- learning_rate:
the contribution of each model to the weights, and it defaults to 1.

Reducing the learning rate will mean the weights will be increased or decreased
to a small degree, forcing the model to train slower
(but sometimes resulting in better performance scores).

- loss:
a loss function to use when updating weights in AdaBoostRegressor.

See also:
- Explaining AdaBoost
http://rob.schapire.net/papers/explaining-adaboost.pdf
"""
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets

# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Create adaboost tree classifier object
ada_boost = AdaBoostClassifier(random_state=0, algorithm="SAMME")

# Train model
model = ada_boost.fit(features, target)

# Make new observations
observations = [[5, 4, 3, 2],
                [2, 3, 4, 5],
                [2, 4, 3, 5],
                [1, 1, 1, 1],
                [2, 2, 2, 2],
                [3, 3, 3, 3]]
# Predict classes
model.predict(observations)
# array([1, 1, 1, 2, 2, 1])
