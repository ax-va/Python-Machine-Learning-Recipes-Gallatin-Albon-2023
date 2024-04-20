"""
Apply a custom transformation to features:
- to np.array
- to pd.DataFrame

An example would be creating a feature that is the natural log of the values of a different feature.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

# Create feature matrix: each row is an observation of the features x_1 and x_2
features = np.array([[2, 3],
                     [2, 3],
                     [2, 3]])
# array([[2, 3],
#        [2, 3],
#        [2, 3]])


# Define some function
def add_ten(x: int) -> int:
    return x + 10


# Create transformer
ten_transformer = FunctionTransformer(add_ten)

# Transform feature matrix
ten_transformer.transform(features)
# array([[12, 13],
#        [12, 13],
#        [12, 13]])

# Alternatively, create the same transformation in pandas using apply.
# Create DataFrame:
df = pd.DataFrame(features, columns=["feature_1", "feature_2"])
#    feature_1  feature_2
# 0          2          3
# 1          2          3
# 2          2          3

# Apply function
df.apply(add_ten)
#    feature_1  feature_2
# 0         12         13
# 1         12         13
# 2         12         13
