"""
Some features are possibly highly correlated.
->
Use a correlation matrix to check for highly correlated features.
->
Leave only one feature from highly correlated features and dpop the other ones that are redundant.

Redundant features can result in an artificially inflated R-squared value
in the case of linear regression or other models.
"""
import pandas as pd
import numpy as np

# Create feature matrix with two highly correlated features
features = np.array([[1, 1, 1],
                     [2, 2, 0],
                     [3, 3, 1],
                     [4, 4, 0],
                     [5, 5, 1],
                     [6, 6, 0],
                     [7, 7, 1],
                     [8, 7, 0],
                     [9, 7, 1]])

# Convert feature matrix into DataFrame
df = pd.DataFrame(features)
#    0  1  2
# 0  1  1  1
# 1  2  2  0
# 2  3  3  1
# 3  4  4  0
# 4  5  5  1
# 5  6  6  0
# 6  7  7  1
# 7  8  7  0
# 8  9  7  1

df.corr()
#           0         1         2
# 0  1.000000  0.976103  0.000000
# 1  0.976103  1.000000 -0.034503
# 2  0.000000 -0.034503  1.000000

df.corr().abs()
#           0         1         2
# 0  1.000000  0.976103  0.000000
# 1  0.976103  1.000000  0.034503
# 2  0.000000  0.034503  1.000000

# Create correlation matrix
corr_matrix = df.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(
    np.triu(
        np.ones(corr_matrix.shape),
        k=1,
    ).astype(bool)
)
#     0         1         2
# 0 NaN  0.976103  0.000000
# 1 NaN       NaN  0.034503
# 2 NaN       NaN       NaN

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
# [1]

# Drop features
df.drop(df.columns[to_drop], axis=1)
#    0  2
# 0  1  1
# 1  2  0
# 2  3  1
# 3  4  0
# 4  5  1
# 5  6  0
# 6  7  1
# 7  8  0
# 8  9  1
