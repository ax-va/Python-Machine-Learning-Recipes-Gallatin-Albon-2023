"""
Know which features are most important in a random forest model.
->
Use the feature_importances_ attribute.

Features with splits that have the greater mean decrease in impurity
(e.g., Gini impurity or entropy in classifiers and variance in regressors)
are considered more important.

The higher the importance score, the more important the feature (all importance scores sum to 1).

Two caveats:

1.
Breaking up nominal categorical features into multiple binary (i.e., one-hot encoded) features
leads to the effect of spreading the importance of that feature across all the binary features.
That can make each feature unimportant even when the original nominal categorical feature is
highly important.

2.
If two features are highly correlated, one feature will claim much of the importance,
making the other feature far less important.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

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

# Create random forest classifier object
random_forest = RandomForestClassifier(random_state=0, n_jobs=-1)
# Train model
model = random_forest.fit(features, target)
# Calculate feature importances of
# ['sepal length (cm)',
#  'sepal width (cm)',
#  'petal length (cm)',
#  'petal width (cm)']

# Get relative importances
importances = model.feature_importances_
# array([0.09090795, 0.02453104, 0.46044474, 0.42411627])

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]
# array([2, 3, 0, 1])

# Rearrange feature names
names = [iris.feature_names[i] for i in indices]
# ['petal length (cm)',
#  'petal width (cm)',
#  'sepal length (cm)',
#  'sepal width (cm)']

# Create plot
plt.figure()
plt.title("Feature Importance")
plt.bar(range(features.shape[1]), importances[indices])
plt.xticks(range(features.shape[1]), names, rotation=45)
plt.tight_layout()
# plt.show()
plt.savefig(f'example-14-07--sklearn--identifying-important-features-in-random-forests--RandomForestClassifier--feature_importances_.svg')
plt.close()
