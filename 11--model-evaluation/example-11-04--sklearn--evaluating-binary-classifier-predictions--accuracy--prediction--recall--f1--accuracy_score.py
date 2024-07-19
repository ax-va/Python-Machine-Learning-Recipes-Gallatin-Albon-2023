"""
Evaluate the quality of a binary classification model.

* Accuracy = num_correct_predictions / num_all_predictions = (TP + TN) / (TP + TN + FP + FN),

where
TP = number of true positives,
TN = number of true negatives,
FP = number of false positives, and
FN = number of false_negatives.

num_false_positive is known as *Type I error*.
num_false_negative is known as *Type II error*.

Accuracy paradox:
When in the presence of imbalanced classes
(e.g., the 99.9% of observations are of class 1 and only 0.1% are class 2),
accuracy suffers from a paradox where a model is highly accurate but lacks predictive power.

* Precision = num_correct_positive_predictions / num_all_positive_predictions = TP / (TP + FP).

In other words, precision answers the question:
how often the positive predictions are correct?

Models with high precision are pessimistic in prediction of the positive class.

* Recall = num_correct_positive_predictions / all_positive_instances = TP / (TP + FN).

In other words, recall answers the question:
how well an ML model can find all instances of the positive class?

Models with high precision are optimistic in prediction of the positive class.

* F_1 = 2 * Precision * Recall / (Precision + Recall).

F_1 answers the question:
of observations labeled as positive, how many are actually positive?

The F_1 score is the harmonic mean, represents a balance between the recall and precision.
"""
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate features matrix and target vector
X, y = make_classification(
    n_samples=10000,
    n_features=3,
    n_informative=3,
    n_redundant=0,
    n_classes=2,
    random_state=1,
)

# Create logistic regression
logreg = LogisticRegression()

# Cross-validate model using accuracy
cross_val_score(logreg, X, y, scoring="accuracy")
# array([0.9555, 0.95  , 0.9585, 0.9555, 0.956 ])

# Cross-validate model using precision
cross_val_score(logreg, X, y, scoring="precision")
# array([0.95963673, 0.94820717, 0.9635996 , 0.96149949, 0.96060606])

# Cross-validate model using recall
cross_val_score(logreg, X, y, scoring="recall")
# array([0.951, 0.952, 0.953, 0.949, 0.951])

# Cross-validate model using F1
cross_val_score(logreg, X, y, scoring="f1")
#  array([0.95529884, 0.9500998 , 0.95827049, 0.95520886, 0.95577889])

# Measure accuracy without cross-validation, create training and test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.1,
    random_state=1
)

# Predict values for training target vector
y_predicted = logreg.fit(X_train, y_train).predict(X_test)

# Calculate accuracy
accuracy_score(y_test, y_predicted)
# 0.947
