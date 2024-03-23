"""
Simulate data to be used for classification.
"""
from sklearn.datasets import make_classification


# Generate features matrix and target vector
features, target = make_classification(
    n_samples=100,
    n_features=3,  # n_informative < n_features => redundant features
    n_informative=3,  # number of features to generate the target vector
    n_redundant=0,
    n_classes=2,
    weights=[.25, .75],  # to simulate datasets with imbalanced classes
    random_state=1,
)

print('Feature Matrix\n', features[:3])
# Feature Matrix
#  [[ 1.06354768 -1.42632219  1.02163151]
#  [ 0.23156977  1.49535261  0.33251578]
#  [ 0.15972951  0.83533515 -0.40869554]]
print('Target Vector\n', target[:3])
# Target Vector
#  [1 0 0]
