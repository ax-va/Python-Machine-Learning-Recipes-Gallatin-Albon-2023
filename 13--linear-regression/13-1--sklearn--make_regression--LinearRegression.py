"""
Train a model that represents a linear relationship between the feature and target vector.

Relationship between a target and only three features is represented as follows:
$$
y = beta_0 + beta_1 * feature_1 + beta_2 * feature_2 + beta_3 * feature_3 + error,
$$
where $beta_0$ is called bias or also intercept.
"""
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

# Generate features matrix, target vector
features, target = make_regression(
    n_samples=100,
    n_features=3,
    n_informative=2,
    n_targets=1,
    noise=0.2,
    coef=False,
    random_state=1,
)

features.shape
# (100, 3)

features
# array([[ 5.85910431e-01,  7.84770651e-01, -9.55425262e-01],
#        [ 7.92806866e-01, -1.23005814e+00,  5.50537496e-01],
#        [-3.49342722e-01, -3.52249846e-01, -1.14251820e+00],
# ...

target.shape
# (100,)

target
# array([ -20.8707476 ,  -22.37241838,  -82.18590239,   16.54439179,
#          -6.22618501,    3.77560239,  -13.56985569,    0.2698249 ,
#           1.68322783,  -16.5074944 ,   57.43169745,   45.80106565,
# ...

# Create linear regression
regression = LinearRegression()

# Fit the linear regression
model = regression.fit(features, target)

# View the intercept
model.intercept_
#  -0.009650118178816669

# View the feature coefficients
model.coef_
# array([1.95531234e-02, 4.42087450e+01, 5.81494563e+01])

# First value in the target vector
target[0]
# -20.870747595269407

# Predict the target value of the first observation
model.predict(features)[0]
# -20.861927709296808

# Get the score of the model on the training data
model.score(features, target)
# 0.9999901732607787
# The score ranges from 0.0 (worst) to 1.0 (best).
# However, evaluating this model is on data it has already seen (the training data).
# Nonetheless, such a high score would be good for our model in a real setting.
