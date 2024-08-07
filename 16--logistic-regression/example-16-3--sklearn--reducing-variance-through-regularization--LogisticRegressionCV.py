"""
Reduce the variance of a logistic regression model.
->
Tune the inverse of the regularization strength hyperparameter, $C$.

A penalty term is added to the loss function,
typically the L1 and L2 penalties.

- L1:
$$\alpha \sum_{j=1}^p |\hat{\beta}_j|$$,

- L2:
$$\alpha \sum_{j=1}^p \hat{\beta}_j^2$$,
where
$\hat{\beta}_j$ is the $j$th of the $p$ features being learned,
$\alpha$ is the regularization strength.

Scikit-Learn follows is using the inverse of the regularization
strength $C = 1 / \alpha$ instead of $\alpha$.
"""
from sklearn.linear_model import LogisticRegressionCV
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Standardize features
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# Create logistic-regression cross-validation
logistic_regression = LogisticRegressionCV(
    penalty='l2',  # Cannot use ['l1', 'l2']
    # Pass to Cs either a list of floats or an integer to generate
    # candidates from a logarithmic scale between 1e-4 and 1e4.
    Cs=10,
    random_state=0,
    n_jobs=-1,
)

# Train model
model = logistic_regression.fit(features_standardized, target)

# Predict
model.predict_proba([[.5, .5, .5, .5]])
# array([[4.82176926e-04, 9.69649759e-01, 2.98680642e-02]])

# logistic regression parameters "intercept_" (3 classes)
model.intercept_
# array([ 0.16098417,  4.90722766, -5.06821183])

# logistic regression parameters "coef_" (3 classes and 4 features)
model.coef_
# array([[-1.95545929,  2.14492589, -4.27975939, -4.05341762],
#        [ 1.55986398, -0.35630069, -1.98088253, -1.64611994],
#        [ 0.39559531, -1.7886252 ,  6.26064191,  5.69953755]])

# the inverse of the regularization strength (one value per class)
model.C_
# array([21.5443469, 21.5443469, 21.5443469])
