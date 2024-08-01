"""
Conduct feature selection on a random forest.
->
Identify the importance features and retrain the model using only the most important features.

Steps:
1. Train a random forest model using all features.
2. Use this model to identify the most important features.
3. Create a new feature matrix that includes only these features.
4. Use the new feature matrix to create a new model.

Two caveats:

1.
Breaking up nominal categorical features into multiple binary (i.e., one-hot encoded) features
leads to the effect of spreading the importance of that feature across all the binary features.
That can make each feature unimportant even when the original nominal categorical feature is
highly important.

2.
If two features are highly correlated, one feature will claim much of the importance,
making the other feature far less important.

See also:
- Variable selection using Random Forests
https://hal.science/file/index/docid/755489/filename/PRLv4.pdf
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.feature_selection import SelectFromModel

# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

iris.feature_names
# ['sepal length (cm)',
#  'sepal width (cm)',
#  'petal length (cm)',
#  'petal width (cm)']
iris.target_names
# array(['setosa', 'versicolor', 'virginica'], dtype='<U10')
features.shape
# (150, 4)
target.shape
# (150,)

# Create random forest classifier
random_forest = RandomForestClassifier(random_state=0, n_jobs=-1)

# Create selector that selects features
# with importance greater than or equal to a threshold.
selector = SelectFromModel(random_forest, threshold=0.3)

# Create new feature matrix using selector
features_important = selector.fit_transform(features, target)
features_important.shape
# (150, 2)
selector.get_feature_names_out()
# array(['x2', 'x3'], dtype=object)

# Train random forest using most important features
model = random_forest.fit(features_important, target)
