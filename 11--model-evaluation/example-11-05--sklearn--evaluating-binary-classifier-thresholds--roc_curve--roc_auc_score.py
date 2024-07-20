"""
Evaluate a binary classifier and various probability thresholds.
->
Use the *receiver operating characteristic* (ROC) curve to evaluate the quality of the binary classifier.

ROC compares the presence of true positives and false positives at every probability threshold
(i.e., the probability at which an observation is predicted to be a class).

By default, scikit-learn predicts an observation is part of the positive class
if the probability is greater than 0.5 (called the *threshold*).

The *true positive rate* (TPR) is the number of observations
correctly predicted as class 1 divided by the number of all class-1 observations:

* TPR = TP / (TP + FN)

The false positive rate (FPR):

* FPR = FP / (FP + TN)

The ROC curve represents the respective TPR and FPR for every probability threshold.
In addition, the ROC curve can also be used as a general metric for a model.
The better a model is, the higher the curve is and thus the greater the area under the curve is.
The area under the ROC curve (AUC ROC) can measure the overall quality of a model at all possible thresholds.
The closer the AUC ROC is to 1, the better the model is.

See also:
- ROC Curves in Python and R
https://community.alteryx.com/t5/Data-Science/ROC-Curves-in-Python-and-R/ba-p/138430

- The Area Under an ROC Curve
https://darwin.unmc.edu/dxtests/roc3.htm
"""
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

# Create feature matrix and target vector
features, target = make_classification(
    n_samples=10000,
    n_features=10,
    n_classes=2,
    n_informative=3,
    random_state=3,
)

# Split into training and test sets
features_train, features_test, target_train, target_test = train_test_split(
features, target, test_size=0.1, random_state=1,
)
features_train.shape[0] / features.shape[0] * 100
# 90.0
features_test.shape[0] / features.shape[0] * 100
# 10.0

# Create classifier
logreg = LogisticRegression()

# Train model
logreg.fit(features_train, target_train)

# Get predicted probabilities for each test observation
# to belong to the negative classe (class 0) and the positive class (class 1)
logreg.predict_proba(features_test)[0:3]
# array([[0.8689235 , 0.1310765 ],
#        [0.46319939, 0.53680061],
#        [0.03395453, 0.96604547]])

# See the classes using classes_
logreg.classes_
# array([0, 1])

# Get predicted probabilities for each test observation to belong to the positive class (class 1)
class_one_target_probabilities = logreg.predict_proba(features_test)[:, 1]
# array([1.31076496e-01, 5.36800611e-01, 9.66045472e-01, 6.03894468e-01,
#        3.33269458e-01, 9.29490744e-01, 3.18894653e-02, 1.55993468e-01,
#        6.70046704e-01, 8.73367618e-02, 8.30450135e-01, 8.68970439e-02,
#        3.33788427e-01, 3.63528737e-02, 6.81767111e-01, 1.73348452e-01,
#        7.22126164e-01, 4.87310400e-01, 2.61622550e-02, 1.16254225e-02,
# ..
#        9.83200028e-01, 1.77050172e-01, 5.67962068e-01, 2.69713044e-01,
#        9.11175746e-01, 1.02330094e-02, 6.34765653e-01, 8.35048054e-01,
#        5.33323416e-01, 5.94873397e-01, 5.61945272e-01, 8.20818686e-02,
#        2.91217635e-02, 9.78411636e-01, 6.10003660e-01, 9.78929918e-01,
#        5.30000914e-02, 5.32801930e-01, 3.25482746e-01, 8.39024835e-01])

# Create true and false positive rates
false_positive_rate, true_positive_rate, threshold = roc_curve(
    target_test, class_one_target_probabilities,
)

# Plot ROC curve
plt.title("Receiver Operating Characteristic (ROC)")
plt.plot(false_positive_rate, true_positive_rate)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0], c=".7"), plt.plot([1, 1], c=".7")  # diagonal line
plt.ylabel("True Positive Rate (TPR)")
plt.xlabel("False Positive Rate (FPR)")
# plt.show()
plt.savefig('example-11-05--sklearn--evaluating-binary-classifier-thresholds--roc_curve--roc_auc_score.svg')
plt.close()
# A classifier that predicts every observation correctly would look like the
# solid line in the ROC output in the figure, going straight up to the top immediately.
# A classifier that predicts at random will appear as the diagonal line.

# For example, a threshold of ~0.50 has a TPR of ~0.83 and an FPR of ~0.16:
print("Threshold:", threshold[124])
# Threshold: 0.4981509475208573
print("True Positive Rate (TPR):", true_positive_rate[124])
# True Positive Rate (TPR): 0.8367346938775511
print("False Positive Rate (FPR):", false_positive_rate[124])
# False Positive Rate (FPR): 0.1627450980392157

# Increase the threshold to ~80% to be predicted as positive (class 1)
print("Threshold:", threshold[49])
# Threshold: 0.8058635463651345
print("True Positive Rate (TPR):", true_positive_rate[49])
# True Positive Rate (TPR): 0.5653061224489796
print("False Positive Rate (FPR):", false_positive_rate[49])
# False Positive Rate (FPR): 0.052941176470588235

# Calculate the area under the ROC curve (AUC ROC)
roc_auc_score(target_test, class_one_target_probabilities)
# 0.9073429371748699
