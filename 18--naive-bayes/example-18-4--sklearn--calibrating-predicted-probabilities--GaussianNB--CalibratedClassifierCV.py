"""
Calibrate the predicted probabilities from naive Bayes classifiers, so they are interpretable.
->
Use CalibratedClassifierCV, in which the training sets are used to train the model,
and the test set is used to calibrate the predicted probabilities.

CalibratedClassifierCV offers two calibration methods:
- Platt's sigmoid model;
- isotonic regression—defined that tends to overfit
when sample sizes are very small (e.g., 100 observations).
"""
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV

# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Create Gaussian naive Bayes object
classifier = GaussianNB()
# Create calibrated k-fold cross-validation with sigmoid calibration
classifier_sigmoid = CalibratedClassifierCV(
    classifier,
    cv=2,  # number of folds
    method='sigmoid',  # Platt's sigmoid model
)
# Calibrate probabilities
classifier_sigmoid.fit(features, target)

# Create new observation
new_observation = [[2.6, 2.6, 2.6, 0.4]]
# View calibrated probabilities
classifier_sigmoid.predict_proba(new_observation)
# array([[0.31859971, 0.63663451, 0.04476578]])

# Demonstrate the significant difference between raw
# and well-calibrated predicted probabilities.
classifier.fit(features, target).predict_proba(new_observation)
# array([[2.31548432e-04, 9.99768128e-01, 3.23532277e-07]])

# While the ranging is the same, the values are very different.
# For the non-calibrated probabilities, the values are very extreme and tends to 0 or 1.
