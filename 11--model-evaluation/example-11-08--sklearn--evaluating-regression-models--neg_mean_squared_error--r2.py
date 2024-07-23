"""
Evaluate the performance of a regression model.
->
Use mean squared error (MSE) or R^2.

1) $MSE = 1 / n \sum_i^n ({y_predicted}_i - {y_true}_i)$

Squaring penalizes a few large errors more than many small errors, for example:
MSE of model A: 0^2 + 10^2 = 100
MSE of model B: 5^2 + 5^2 = 50

MSE would consider model A (MSE = 100) worse than model B (MSE = 50).
In practice, this implication is rarely an issue,
and MSE works perfectly fine as an evaluation metric.

In Scikit-Learn, arguments of the scoring parameter
assume that higher values are better than lower values.
->
In Scikit-Learn, the negative MSE is used with scoring='neg_mean_squared_error'.

2) $R^2 = 1 - \sum_i^n ({y_true}_i - {y_predicted}_i) / \sum_i^2 ({y_true}_i - y_mean)$

The closer that $R^2$ is to 1, the better the model.
"""
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

# Generate features matrix, target vector
features, target = make_regression(
    n_samples=100,
    n_features=3,
    n_informative=3,
    n_targets=1,
    noise=50,
    coef=False,
    random_state=1,
)

# Create a linear regression object
linreg = LinearRegression()

# Cross-validate the linear regression using (negative) MSE
cross_val_score(linreg, features, target, scoring='neg_mean_squared_error')
# array([-1974.65337976, -2004.54137625, -3935.19355723, -1060.04361386,
#        -1598.74104702])

# Cross-validate the linear regression using R-squared
cross_val_score(linreg, features, target, scoring='r2')
# array([0.8622399 , 0.85838075, 0.74723548, 0.91354743, 0.84469331])
