"""
Evaluate the model's performance that predicts three or more classes.
->
Use cross-validation with appropriate evaluation metrics for multiclass classification.

Metrics:
- accuracy for balanced classes (i.e., a roughly equal number of observations in each class of the target vector)
- precision and F1 score (treating our data as a set of binary classes) for imbalanced classes

In Scikit-Learn, there are the following options to average the evaluation scores:
- macro: Calculate the mean of metric scores for each class, weighting each class equally.
- weighted: Calculate the mean of metric scores for each class, weighting each class
            proportional to its size in the data.
- micro: Calculate the mean of metric scores for each observation-class combination.
"""
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Generate features matrix and target vector
features, target = make_classification(
    n_samples=10000,
    n_features=3,
    n_informative=3,
    n_redundant=0,
    n_classes=3,
    random_state=1,
)

# Create logistic regression
logreg = LogisticRegression()

# Cross-validate model using accuracy
cross_val_score(logreg, features, target, scoring='accuracy')
# array([0.8405, 0.829 , 0.827 , 0.8155, 0.8205])

# Cross-validate model using macro averaged precision score
cross_val_score(logreg, features, target, scoring='precision_macro')
# array([0.8403544 , 0.82892836, 0.82667594, 0.81510466, 0.82041164])

# "macro" refers to the method used to average the evaluation scores from the classes.
# The options are macro, weighted, and micro.

# Cross-validate model using macro averaged F1 score
cross_val_score(logreg, features, target, scoring='f1_macro')
# array([0.84012014, 0.82895312, 0.82675308, 0.81515121, 0.82042629])

# Cross-validate model using weighted averaged F1 score
cross_val_score(logreg, features, target, scoring='f1_weighted')
# array([0.84013913, 0.8289688 , 0.82680304, 0.8151928 , 0.82048321])

# Cross-validate model using micro averaged F1 score
cross_val_score(logreg, features, target, scoring='f1_micro')
# array([0.8405, 0.829 , 0.827 , 0.8155, 0.8205])
