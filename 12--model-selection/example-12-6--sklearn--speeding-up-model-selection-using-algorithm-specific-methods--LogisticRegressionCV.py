"""
Speed up model selection without using additional compute power if you just select hyperparameters.
->
In Scikit-Learn, many learning algorithms (e.g., ridge, lasso, and elastic net regression)
have an algorithm-specific cross-validation method to take advantage of this.
Use Scikit-Learn's model-specific cross-validation hyperparameter tuning: LogisticRegressionCV, etc.

See also:
- LogisticRegressionCV
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html

- Model specific cross-validation
https://scikit-learn.org/stable/modules/grid_search.html#model-specific-cross-validation
"""
from sklearn import linear_model, datasets

# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Create cross-validated logistic regression
logreg_cv = linear_model.LogisticRegressionCV(
    Cs=100,  # Use 100 candidate values for C in a logarithmic scale between 1e-4 and 1e4
    max_iter=500,
    solver='liblinear',
)

# Train model
logreg_cv.fit(features, target)
