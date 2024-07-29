"""
Visualize a model created by a decision tree learning algorithm.
->
Export the decision tree model into DOT format, then visualize,
for both decision tree classifier and regressor.

Precondition: GraphViz is installed.

Install GraphViz on Ubuntu:
$ sudo dnf install graphviz

For example, in the root decision rule,
"gini = 0.667" means that the Gini index equals 0.667,
"samples = 150" means that the number of observations equals 150, and
"value = [50, 50, 50]" means that the number of observations in each class is 50.
"""
import pydotplus
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from IPython.display import Image
from sklearn import tree

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

# Create decision tree classifier object
decision_tree = DecisionTreeClassifier(random_state=0)

# Train model
model = decision_tree.fit(features, target)

# Create DOT data
dot_data = tree.export_graphviz(
    decision_tree,
    out_file=None,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
)

# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)

# Export into files
graph.write_svg("example-14-03--pydotplus--visualizing-decision-tree-model.svg")
# graph.write_png("example-14-03--pydotplus--visualizing-decision-tree-model.png")
# graph.write_pdf("example-14-03--pydotplus--visualizing-decision-tree-model.pdf")

# Comment out in SVG this part:
"""
<g id="node18" class="node">
    <title>\n</title>
    <polygon fill="none" stroke="black" points="682,-639.5 628,-639.5 628,-603.5 682,-603.5 682,-639.5"/>
</g>
"""

# # Show graph in JupyterLab
# Image(graph.create_png())
