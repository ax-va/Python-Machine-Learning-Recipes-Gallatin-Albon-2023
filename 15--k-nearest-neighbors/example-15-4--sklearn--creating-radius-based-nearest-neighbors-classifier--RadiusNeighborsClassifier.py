"""
Predict a class of an unknown observation based on classes of all observations within a certain distance.
->
Use Scikit-Learn RadiusNeighborsClassifier.

In the radius-based nearest neighbor (RNN) classification, an observation's class
is predicted from the classes of all observations within a given radius.
The radius is a hyperparameter that can be tuned during model selection.
"""
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Create standardizer
standardizer = StandardScaler()

# Standardize features
features_standardized = standardizer.fit_transform(features)

# Train a radius neighbors classifier
rnn = RadiusNeighborsClassifier(
    radius=.5,
    n_jobs=-1,
    outlier_label=-1,  # class -1 for outliers
).fit(features_standardized, target)

# Create new observations
new_observations = [[.75, .75, .75, .75],
                    [.5, .5, .5, .5],
                    [1., 1., 1., 1.],
                    [100., 100., 100., 100.]]  # definitive outlier assigned to class -1

# Predict the class of two observations
rnn.predict(new_observations)
# UserWarning: Outlier label -1 is not in training classes. All class probabilities of outliers will be assigned with 0.
#  array([1, 1, 2, -1])
