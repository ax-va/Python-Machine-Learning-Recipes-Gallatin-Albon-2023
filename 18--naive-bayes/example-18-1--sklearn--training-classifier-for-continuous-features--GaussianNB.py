"""
Train a naive Bayes classifier with only continuous features.
->
Use a Gaussian naive Bayes classifier in Scikit-Learn.

Warning:
    Found out that "GaussianNB(priors=...)" does not set priors.

The Gaussian naive Bayes classifier is best used in cases of all continuous features.
The likelihood of the feature values $x_j$, given an observation of class $y$,
follows a normal distribution:
$$
p(x_j | y) = 1 / \sqrt{ 2 \pi \sigma_{x_jy}^2 } \exp{ -(x_j - \mu_{x_jy})^2 / (2 sigma_{x_jy}^2) }
$$,
where $\sigma_{x_jy}^2$ and $\mu_{x_jy}$ are the variance and mean of $x_j$ for $y$.

Then
$$
p(y | observations) ~ p(observations | y) p(y) = \prod_{observation} p(observation | y) p(y)
~ \prod_{observation} \prod_{x_j} p(x_j | y) p(y)
$$.
The formula will be logarithmed.

Notice:
Predicted probabilities obtained by "predict_proba" is are not calibrated.
->
That is, they should not be believed.

See also:
- How Naive Bayes classifier algorithm works in machine learning
https://dataaspirant.com/naive-bayes-classifier-machine-learning/
"""
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Create Gaussian naive Bayes
classifer = GaussianNB()  # The prior is adjusted based on the data
# Train model
model = classifer.fit(features, target)
# Create new observation
new_observation = [[5.2, 3.6, 1.5, 0.3]]
# Predict class
model.predict(new_observation)
# array([0])

# Set prior probabilities p(y) of each class of 3 (does not work)
clf = GaussianNB(priors=[1e-12, 1-1e-11, 9e-12])
# Train model
model_priors = clf.fit(features, target)
# Create new observation
new_observation = [[5.2, 3.6, 1.5, 0.3]]
# Predict class
model_priors.predict(new_observation)
# array([0])
clf.predict(new_observation)
