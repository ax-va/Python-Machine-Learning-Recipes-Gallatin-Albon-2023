"""
Know the predicted class probabilities for an observation.
->
Set "probability=True" to Scikit-Learn's SVC, train the model, then use predict_proba.

In an SVC with two classes, *Platt scaling* can be used,
wherein first the SVC is trained, and then a separate cross-validated
logistic regression is trained to map the SVC outputs into probabilities:
$$
P(y = 1 | x) = 1 / (1 + \exp{a \times d_x + b})
$$,
where $a$, $b$ are parameter vectors, and
$d_x$ is the signed distance of $x$ from the hyperplane.

For more than two classes, an extension of Platt scaling is used.

Because of the composition of the learning algorithms and cross validation,
predicted probabilities might not always match the predicted classes.
"""
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Standardize features
features_standardized = StandardScaler().fit_transform(features)

# Create support vector classifier object
svc = SVC(kernel="linear", probability=True, random_state=0)
# Train classifier
model = svc.fit(features_standardized, target)

# Create new observation
new_observation = [[.4, .4, .4, .4]]
# View predicted probabilities
model.predict_proba(new_observation)
# array([[0.00541761, 0.97348825, 0.02109414]])
