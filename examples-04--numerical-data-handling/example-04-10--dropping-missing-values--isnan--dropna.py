"""
Delete observations containing missing values using
- NumPy's isnan
- Pandas' dropna

Deleting observations can introduce bias into our data.

There are three types of missing data:

1. Missing completely at random (MCAR)
Some random event (for example, if six was thrown randomly) => skip the feature value

2. Missing at random (MAR)
Some feature is known (for example, gender) => randomly skip another feature value (for example, salary)

3. Missing not at random (MNAR)
Some feature value was skipped for an unknown reason

If the value is MNAR, the fact that a value is missing is itself information.
Deleting that can inject bias into the data.
"""
import numpy as np
import pandas as pd

# Create feature matrix
features = np.array([[1.1, 11.1],
                     [2.2, 22.2],
                     [3.3, 33.3],
                     [4.4, 44.4],
                     [np.nan, 55.5]])
# array([[ 1.1, 11.1],
#        [ 2.2, 22.2],
#        [ 3.3, 33.3],
#        [ 4.4, 44.4],
#        [ nan, 55.5]])


# Keep only observations that are not (denoted by ~) missing
features[~np.isnan(features).any(axis=1)]
# array([[ 1.1, 11.1],
#        [ 2.2, 22.2],
#        [ 3.3, 33.3],
#        [ 4.4, 44.4]])

# Alternatively

df = pd.DataFrame(features, columns=["feature_1", "feature_2"])
#    feature_1  feature_2
# 0        1.1       11.1
# 1        2.2       22.2
# 2        3.3       33.3
# 3        4.4       44.4
# 4        NaN       55.5

# Drop observations with missing values
df.dropna()
#    feature_1  feature_2
# 0        1.1       11.1
# 1        2.2       22.2
# 2        3.3       33.3
# 3        4.4       44.4
