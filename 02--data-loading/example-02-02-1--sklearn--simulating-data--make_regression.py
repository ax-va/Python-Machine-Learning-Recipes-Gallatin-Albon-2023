"""
Simulate data to be used with linear regression.
"""
from sklearn.datasets import make_regression


# Generate features matrix, target vector, and the true coefficients
features, target, coefficients = make_regression(
    n_samples=100,
    n_features=3,  # n_informative < n_features => redundant features
    n_informative=3,  # number of features to generate the target vector
    n_targets=1,
    noise=0.0,
    coef=True,
    random_state=1,
)

print('Feature Matrix\n', features[:3])
# Feature Matrix
#  [[ 1.29322588 -0.61736206 -0.11044703]
#  [-2.793085    0.36633201  1.93752881]
#  [ 0.80186103 -0.18656977  0.0465673 ]]
print('Target Vector\n', target[:3])
# Target Vector
#  [-10.37865986  25.5124503   19.67705609]
