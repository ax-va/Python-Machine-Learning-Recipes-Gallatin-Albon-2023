"""
Understand how the performance of a model changes as the value of some hyperparameter changes.
->
Plot the hyperparameter against the model accuracy (validation curve).

See also:
- Scikit-Learn: Validation Curve
https://scikit-learn.org/stable/modules/learning_curve.html#validation-curve
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import validation_curve

# Load data
digits = load_digits()
# Create feature matrix and target vector
features, target = digits.data, digits.target
# Create range of values for parameter
param_range = np.arange(1, 250, 2)
param_range.shape
# (125,)
param_range
# array([  1,   3,   5,   7,   9,  11,  13,  15,  17,  19,  21,  23,  25,
#         27,  29,  31,  33,  35,  37,  39,  41,  43,  45,  47,  49,  51,
# ...
#        209, 211, 213, 215, 217, 219, 221, 223, 225, 227, 229, 231, 233,
#        235, 237, 239, 241, 243, 245, 247, 249])

# Calculate accuracy on training and test set using range of parameter values
train_scores, test_scores = validation_curve(
    RandomForestClassifier(),  # classifier
    features,  # feature matrix
    target,  # target vector
    param_name="n_estimators",  # hyperparameter to examine
    param_range=param_range,  # range of hyperparameter's values
    cv=3,  # number of folds for cross-validation
    scoring="accuracy",  # performance metric
    n_jobs=-1,  # all computer cores to use
)
train_scores.shape
# (125, 3)
test_scores.shape
# (125, 3)

# Calculate mean and standard deviation for training set scores
train_mean = np.mean(train_scores, axis=1)  # mean in each row: going along axis 1 for each element fixed in axis 0
train_std = np.std(train_scores, axis=1)  # std in each row: going along axis 1 for each element fixed in axis 0

# Calculate mean and standard deviation for test set scores
test_mean = np.mean(test_scores, axis=1)  # mean in each row: going along axis 1 for each element fixed in axis 0
test_std = np.std(test_scores, axis=1)  # std in each row: going along axis 1 for each element fixed in axis 0

# """
# |-------> axis 1
# |  1  2
# |  4  5
# |  7  8
# V
# axis 0
# """

# Plot mean accuracy scores for training and test sets
plt.plot(
    param_range, train_mean, "--",
    label="Training score",
    color="black",
)
plt.plot(
    param_range, test_mean,
    label="Test cross-validation score",
    color="dimgrey",
)

# Plot accuracy bands for training and test sets
plt.fill_between(
    param_range,
    train_mean - train_std,
    train_mean + train_std,
    color="gray",
)
plt.fill_between(
    param_range,
    test_mean - test_std,
    test_mean + test_std,
    color="gainsboro",
)

# Create plot
plt.title("Validation Curve with Random Forest")
plt.xlabel("Number of Trees")
plt.ylabel("Accuracy Score")
plt.tight_layout()
plt.legend(loc="best")
# plt.show()
plt.savefig('example-11-13--visualizing-effect-of-hyperparameter-values--sklearn--validation_curve--matplotlib--fill_between.svg')
plt.close()
