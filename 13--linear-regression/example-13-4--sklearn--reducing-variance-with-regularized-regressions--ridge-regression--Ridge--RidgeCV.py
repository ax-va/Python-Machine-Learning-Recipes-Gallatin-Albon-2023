"""
Reduce the variance of your linear regression model.
->
Use a learning algorithm that includes a shrinkage penalty
(also called regularization) like ridge regression and lasso regression.

Regularization is a method of penalizing complex models to reduce their variance.

In the standard linear regression, we minimize the residual sum of squares (RSS)
$$
RSS = \sum_{i=1}^n (y_{true, i} - y_{predicted, i})
$$.

In the regularized regression, we minimize RSS and some penalty
(called a shrinkage penalty) for the total size of the coefficient values.

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
"""
from sklearn.linear_model import Ridge, RidgeCV
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

# Create ridge regression with an alpha value
regression = Ridge(alpha=0.5)
# Fit the linear regression
model = regression.fit(features_standardized, target)

# RidgeCV allows us to select the ideal value for alpha.
# Create ridge regression with three alpha values:
regr_cv = RidgeCV(alphas=[0.1, 1.0, 10.0])
# Fit the linear regression
model_cv = regr_cv.fit(features_standardized, target)

# View coefficients
model_cv.coef_
# array([1.29223201e-02, 4.40972291e+01, 5.38979372e+01])

# View the best alpha
model_cv.alpha_
# 0.1
