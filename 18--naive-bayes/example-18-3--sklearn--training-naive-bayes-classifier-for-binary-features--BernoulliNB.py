"""
Train a naive Bayes classifier for binary features, e.g.,
a nominal categorical feature that has been one-hot encoded.
->
Use a Bernoulli naive Bayes classifier.

Bernoulli naive Bayes is often used in text classification,
when our feature matrix is simply the presence or absence
of a word in a document.
"""
import numpy as np
from sklearn.naive_bayes import BernoulliNB

# Create three binary features
features = np.random.randint(2, size=(100, 3))
# array([[1, 0, 1],
#        [0, 1, 1],
#        [0, 1, 0],
# ...
#        [1, 0, 0],
#        [1, 0, 1],
#        [0, 0, 0]])

# Create a binary target vector
target = np.random.randint(2, size=(100, 1)).ravel()
# array([1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1,
# ...
#        1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0])

# Create Bernoulli naive Bayes with prior probabilities of each class
classifier = BernoulliNB(
    # to set priors for two classes;
    # if not set, they are learned from data
    class_prior=[0.25, 0.5],
    # # to use a uniform distribution for classes
    # class_prior = None,
    # fit_prior=False,
    # # smoothing hyperparameter; "alpha=0.0" -> no smoothing
    # alpha=1.0,
)

# Train model
model = classifier.fit(features, target)
