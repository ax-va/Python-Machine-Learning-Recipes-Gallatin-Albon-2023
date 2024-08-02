"""
Manually determine the structure and size of a decision tree.
->
Use parameters in Scikit-Learn for ree-based learning algorithms:

- max_depth
Maximum depth of the tree.
If None, the tree is grown until all leaves are pure.
If an integer, the tree is effectively "pruned" to that depth.

- min_samples_split
Minimum number of observations at a node before that node is split.
If an integer is passed as an argument, it determines the raw minimum.
If a float is passed, the minimum is the percent of total observations.

- min_samples_leaf
Minimum number of observations required to be at a leaf.
Uses the same arguments as min_samples_split.

- max_leaf_nodes
Maximum number of leaves.

- min_impurity_split
Minimum impurity decrease required before a split is performed.

To get shallower trees (sometimes called *stumps*) that are simpler models
and thus have lower variance, used parameters are max_depth and min_impurity_split.
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets

# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Create decision tree classifier
decision_tree = DecisionTreeClassifier(
    random_state=0,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0,
    max_leaf_nodes=None,
    min_impurity_decrease=0,
)
