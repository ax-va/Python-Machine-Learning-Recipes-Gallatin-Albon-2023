"""
Visually compare the model's quality.
->
Use a confusion matrix, which compares predicted classes and true classes.

See also:
- Scikit-Learn: confusion_matrix
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
"""
# Load libraries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd

# Load data
iris = datasets.load_iris()
# Create features matrix
features = iris.data
# Create target vector
target = iris.target

# Create list of target class names
class_names = iris.target_names
# array(['setosa', 'versicolor', 'virginica'], dtype='<U10')

# Create training and test set
features_train, features_test, target_train, target_test = train_test_split(
features, target, random_state=2
)
features_train.shape[0] / features.shape[0] * 100
# 74.66666666666667
features_test.shape[0] / features.shape[0] * 100
# 25.333333333333336

# Create logistic regression
classifier = LogisticRegression()
# Train model and make predictions
target_predicted = classifier.fit(features_train, target_train).predict(features_test)

# Create confusion matrix
conf_matrix = confusion_matrix(target_test, target_predicted)

# Create pandas dataframe
df_conf_matrix = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)

# Create heatmap
sns.heatmap(df_conf_matrix, annot=True, cbar=None, cmap="Blues")
plt.title("Confusion Matrix")
plt.ylabel("True Class", fontweight='bold')
plt.xlabel("Predicted Class", fontweight='bold')
# plt.show()
plt.savefig('example-11-07--visualizing-classifiers-performance--sklearn--confusion_matrix--seaborn--heatmap.svg')
plt.close()
