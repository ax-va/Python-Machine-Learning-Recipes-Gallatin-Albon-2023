"""
Evaluate how well a classification model generalizes to unforeseen data.
->
Use k-fold cross-validation (KFCV).

validation = observations (features and targets) are split into two sets,
called the training set and the test set

Steps of k-fold cross-validation (KFCV):
1. Split the data into k parts called folds.
2. Then, train the model using k – 1 folds (combined into one training set)
and test it on the remaining fold (used as a test set).
3. Repeat k times with unique training and test folds.
4. Average the performance on the model for each of the k iterations
to produce an overall measurement.

Preprocessing steps:
1. The observations are independent and identically distributed (IID).
->
It is a good idea to shuffle observations when assigning to folds.

2. For a classifier, folds have to contain roughly the same percentage of observations
from each of the different target classes (called stratified k-fold).

In Scikit-Learn, replace KFold class with StratifiedKFold.

3. Standardize the data on the training set,
and apply that transformation to the test set too.

In Scikit-Learn:
----------------------------------------------------
from sklearn.model_selection import train_test_split
# ...
# Create training and test sets
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.1, random_state=1
)

# Fit standardizer to training set
standardizer.fit(features_train)

# Apply to both training and test sets which can then be used to train models
features_train_std = standardizer.transform(features_train)
features_test_std = standardizer.transform(features_test)
----------------------------------------------------

In Scikit-Learn, that preprocessing step is performed using the pipeline package.

See also:
- Why every statistician should know about cross-validation
https://robjhyndman.com/hyndsight/crossvalidation/

- Scikit-Learn: Cross-Validation Gone Wrong
https://betatim.github.io/posts/cross-validation-gone-wrong/
"""
from sklearn import datasets
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load digits dataset
digits = datasets.load_digits()
# Create features matrix
features = digits.data
# Create target vector
target = digits.target

# Create standardizer
standardizer = StandardScaler()
# Create logistic regression object
logreg = LogisticRegression()

# Create a pipeline that standardizes the data, then runs logistic regression
pipeline = make_pipeline(
    standardizer,  # preprocessing step 3 described above
    logreg,  # training a model
)

# Create k-fold cross-validation
kf = KFold(
    n_splits=5,
    shuffle=True,  # Perform shuffling the data
    random_state=0,
)

# Conduct k-fold cross-validation
cv_scores = cross_val_score(
    pipeline,  # pipeline
    features,  # feature matrix
    target,  # target vector
    cv=kf,  # cross-validation technique
    scoring="accuracy",  # metric for success
    n_jobs=-1,  # Use every CPU core available to parallelize computation
)

cv_scores
# array([0.96111111, 0.95833333, 0.97771588, 0.96935933, 0.97214485])

# Calculate mean
cv_scores.mean()
# 0.9677329000309502
