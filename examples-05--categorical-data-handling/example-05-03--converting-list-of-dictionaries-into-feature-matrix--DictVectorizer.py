"""
Convert a list dictionaries into a feature matrix.
"""
import pandas as pd
from sklearn.feature_extraction import DictVectorizer

# Create list of dictionaries
data_dict = [
    {"Red": 2, "Blue": 4},
    {"Red": 4, "Blue": 3},
    {"Red": 1, "Yellow": 2},
    {"Red": 2, "Yellow": 2},
]

# Create dictionary vectorizer
dict_vectorizer = DictVectorizer(sparse=False)

# Convert dictionary to feature matrix
features = dict_vectorizer.fit_transform(data_dict)
# array([[4., 2., 0.],
#        [3., 4., 0.],
#        [0., 1., 2.],
#        [0., 2., 2.]])

# Get feature names
feature_names = dict_vectorizer.get_feature_names_out()
# array(['Blue', 'Red', 'Yellow'], dtype=object)

# Create dataframe from features
pd.DataFrame(features, columns=feature_names)
#    Blue  Red  Yellow
# 0   4.0  2.0     0.0
# 1   3.0  4.0     0.0
# 2   0.0  1.0     2.0
# 3   0.0  2.0     2.0
