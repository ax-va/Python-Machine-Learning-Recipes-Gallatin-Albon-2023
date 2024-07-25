"""
Get a quick description of a classifier’s performance.
->
Use Scikit-Learn's classification_report.

That includes precision, recall, F_1 score, and support that is
referred to the number of observations in each class in the test data.
"""
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load data
iris = datasets.load_iris()
# Create features matrix
features = iris.data
features.shape
# (150, 4)  # 150 observations, 4 features
# Create target vector
target = iris.target
target.shape
# (150, )
# Create list of target class names
class_names = iris.target_names
# array(['setosa', 'versicolor', 'virginica'], dtype='<U10')  # 3 classes: 0, 1, 2

# Create training and test set of ~75½ and ~25% of data, respectively
features_train, features_test, target_train, target_test = train_test_split(
features, target,
    random_state=0,
)
target_train.shape[0] / target.shape[0] * 100
# 74.66666666666667
target_test.shape[0] / target.shape[0] * 100
# 25.333333333333336

# Create logistic regression
classifier = LogisticRegression()
# Train model and make predictions
model = classifier.fit(features_train, target_train)
target_predicted = model.predict(features_test)

# Create a classification report
print(
    classification_report(
        target_test, target_predicted,
        target_names=class_names,
    )
)
#               precision    recall  f1-score   support
#
#       setosa       1.00      1.00      1.00        13
#   versicolor       1.00      0.94      0.97        16
#    virginica       0.90      1.00      0.95         9
#
#     accuracy                           0.97        38
#    macro avg       0.97      0.98      0.97        38
# weighted avg       0.98      0.97      0.97        38

# "support" means
for i in range(3):
    print(target_test[target_test==i].shape[0])
# 13
# 16
# 9
target_test.shape[0]
# 38
