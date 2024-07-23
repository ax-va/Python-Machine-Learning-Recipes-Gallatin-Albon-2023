"""
Evaluate a model using a custom metric.
->
Create the metric as a function and convert it into a scorer function using Scikit-Learn's make_scorer

See also:
- Sckit-Learn: make_scorer
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
"""
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.datasets import make_regression

# Generate features matrix and target vector
features, target = make_regression(
    n_samples=100,
    n_features=3,
    random_state=1,
)

# Create training set and test set
features_train, features_test, target_train, target_test = train_test_split(
features, target, test_size=0.10, random_state=1
)


# Create custom metric
def custom_metric(target_test, target_predicted):
    # Calculate R-squared score
    r2 = r2_score(target_test, target_predicted)
    # Return R-squared score
    return r2


# Make scorer and define that higher scores are better
custom_score = make_scorer(custom_metric, greater_is_better=True)

# Create ridge regression object
classifier = Ridge()
# Train ridge regression model
model = classifier.fit(features_train, target_train)
# Apply custom scorer
custom_score(model, features_test, target_test)
# 0.9997906102882058

# Compare custom_metric and r2
target_predicted = model.predict(features_test)
r2_score(target_test, target_predicted)
