"""
One-hot encode nominal features.

One-hot encoding (in machine learning literature)
or dummying (in statistical and research literature)
excludes ordering that is not present in the features.

Recommended: after one-hot encoding a feature, drop one of the one-hot
encoded features in the resulting matrix to avoid linear dependence
https://stats.stackexchange.com/questions/231285/dropping-one-of-the-columns-when-using-one-hot-encoding
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer

# Create feature
feature = np.array([["Texas"],
                    ["California"],
                    ["Texas"],
                    ["Delaware"],
                    ["Texas"]])
# array([['Texas'],
#        ['California'],
#        ['Texas'],
#        ['Delaware'],
#        ['Texas']], dtype='<U10')

# Create one-hot encoder
one_hot = LabelBinarizer()

# One-hot encode feature
one_hot.fit_transform(feature)
# array([[0, 0, 1],
#        [1, 0, 0],
#        [0, 0, 1],
#        [0, 1, 0],
#        [0, 0, 1]])

# View feature classes
one_hot.classes_
# array(['California', 'Delaware', 'Texas'], dtype='<U10')

# Reverse one-hot encoding
one_hot.inverse_transform(one_hot.transform(feature))
# array(['Texas', 'California', 'Texas', 'Delaware', 'Texas'], dtype='<U10')

# Create dummy variables from feature
pd.get_dummies(feature[:, 0])
#    California  Delaware  Texas
# 0       False     False   True
# 1        True     False  False
# 2       False     False   True
# 3       False      True  False
# 4       False     False   True

# Create multiclass feature
multiclass_feature = [("Texas", "Florida"),
                      ("California", "Alabama"),
                      ("Texas", "Florida"),
                      ("Delaware", "Florida"),
                      ("Texas", "Alabama")]
# [('Texas', 'Florida'),
#  ('California', 'Alabama'),
#  ('Texas', 'Florida'),
#  ('Delaware', 'Florida'),
#  ('Texas', 'Alabama')]

# Create multiclass one-hot encoder
one_hot_multiclass = MultiLabelBinarizer()

# One-hot encode multiclass feature
one_hot_multiclass.fit_transform(multiclass_feature)
# array([[0, 0, 0, 1, 1],
#        [1, 1, 0, 0, 0],
#        [0, 0, 0, 1, 1],
#        [0, 0, 1, 1, 0],
#        [1, 0, 0, 0, 1]])

# View classes
one_hot_multiclass.classes_
# array(['Alabama', 'California', 'Delaware', 'Florida', 'Texas'],
#       dtype=object)
