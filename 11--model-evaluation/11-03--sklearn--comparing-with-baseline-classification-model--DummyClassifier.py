"""
Create a simple baseline classifier to compare against another model.
->
Use DummyClassifier to create a simple model to use as a baseline that is a random guessing.

See also:
- Scikit-Learn: DummyClassifier
https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
"""
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data
iris = load_iris()

# Create target vector and feature matrix
features, target = iris.data, iris.target

# Split into training and test set
features_train, features_test, target_train, target_test = train_test_split(
features, target, random_state=0
)
features_train.shape[0] / features.shape[0] * 100
# 74.66666666666667
features_test.shape[0] / features.shape[0] * 100
# 25.333333333333336

# Create dummy classifier
dummy = DummyClassifier(
    strategy='uniform',  # "stratified" or "uniform"
    random_state=1,
)
# Examples:
# strategy="stratified" ->
# If 20% of the observations in the training data are women,
# then DummyClassifier will predict women 20% of the time.
# strategy="uniform" ->
# If 20% of observations are women and 80% are # men,
# uniform will produce predictions that are 50% women and 50% men.

# "Train" model
dummy.fit(features_train, target_train)
# Get accuracy score
dummy.score(features_test, target_test)
# 0.42105263157894735

# Compare the dummy classifier with another model
# Create classifier
classifier = RandomForestClassifier()
# Train model
classifier.fit(features_train, target_train)
# Get accuracy score
classifier.score(features_test, target_test)
# 0.9736842105263158
