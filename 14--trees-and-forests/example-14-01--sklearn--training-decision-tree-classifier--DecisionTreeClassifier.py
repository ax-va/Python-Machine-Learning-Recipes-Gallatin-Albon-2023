"""
Train a classifier using a decision tree.
->
Use Scikit-Learn's DecisionTreeClassifier.

Decision tree learners attempt to find a decision rule
that produces the greatest decrease in impurity at a node.

By default, DecisionTreeClassifier uses Gini impurity:
$$
G(t) = 1 - \sum_{i=1}^c p_i^2
$$,
where
$G(t)$ is the Gini impurity at node $t$, and
$p_i$ is the proportion of observations of class $c$ at node $t$.

See also:
- Decision Tree Learning
https://www.cs.princeton.edu/courses/archive/spr07/cos424/papers/mitchell-dectrees.pdf
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets

# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target
iris.target_names
# array(['setosa', 'versicolor', 'virginica'], dtype='<U10')
features.shape
# (150, 4)
target.shape
# (150,)

# Create decision tree classifier
decision_tree = DecisionTreeClassifier(random_state=0)
# DecisionTreeClassifier(random_state=0)

# Train model
model = decision_tree.fit(features, target)
# DecisionTreeClassifier(random_state=0)

# Make new observation
observation = [[5, 4, 3, 2]]
# Predict class
model.predict(observation)
# array([1])

# See predicted class probabilities for the three classes
model.predict_proba(observation)
# array([[0., 1., 0.]])

# Use a different impurity measurement
decision_tree_entropy = DecisionTreeClassifier(
    criterion='entropy',
    random_state=0,
)
# Train model
model_entropy = decision_tree_entropy.fit(features, target)
# Predict class
model_entropy.predict(observation)
# array([1])
