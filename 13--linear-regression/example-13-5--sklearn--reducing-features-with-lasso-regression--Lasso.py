"""
Simplify your linear regression model by reducing the number of features.

In ridge regression, we minimize
$$
RSS + \alpha \sum_{j=1}^p beta_j^2
$$,
where beta_j for j=1,...,p are feature coefficients and \alpha is a hyperparameter.

In lasso regression, we minimize
$$
RSS/(2n) + \alpha \sum_{j=1}^p \abs{beta_j}
$$,
where n is the number of observations.

A general rule of thumb:
ridge regression often produces slightly better predictions than lasso,
but lasso produces more interpretable models.

elastic net = a regression model with both penalties included

One final note:
In linear regression, the value of the coefficients is partially determined by the scale of the feature,
and in regularized models, all coefficients are summed together.
->
Make sure to standardize the feature prior to training.

Lasso regression's penalty can shrink the coefficients of a model to zero,
effectively reducing the number of features in the model.
->
Reduce variance while improving the interpretability of our model
since fewer features are easier to explain.
"""
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
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
features[:3]
# array([[ 0.58591043,  0.78477065, -0.95542526],
#        [ 0.79280687, -1.23005814,  0.5505375 ],
#        [-0.34934272, -0.35224985, -1.1425182 ]])

# Standardize features
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)
features_standardized[:3]
# array([[ 0.51962033,  0.76789921, -1.12310414],
#        [ 0.74965359, -1.24970484,  0.49985199],
#        [-0.52022036, -0.37068744, -1.32473172]])

# Create lasso regression with alpha value
regression = Lasso(alpha=0.5)
# Fit the linear regression
model = regression.fit(features_standardized, target)

# Lasso regression's penalty can shrink the coefficients of a model to zero,
# effectively reducing the number of features in the model.

# View coefficients
model.coef_
# array([-0.        , 43.58618393, 53.39523724])

# -> beta_1 in beta_1 * x_1 + beta_2 * x_2 + beta_3 * x_3 is not used in the model with alpha=0.5

# Create lasso regression with a high alpha
regression_a10 = Lasso(alpha=10)
model_a10 = regression_a10.fit(features_standardized, target)
model_a10.coef_
# array([-0.        , 32.92181899, 42.73086731])

regression_a100 = Lasso(alpha=100)
model_a100 = regression_a100.fit(features_standardized, target)
model_a100.coef_
# array([-0.,  0.,  0.])
