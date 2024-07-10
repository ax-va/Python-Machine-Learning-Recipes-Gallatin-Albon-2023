"""
Remove uninformative features in a categorical target vector.
->
1. For categorical features, calculate a chi-square statistic between each feature and the target vector.
2. For quantitative features, compute the ANOVA F-value between each feature and the target vector.

1. Chi-squared statistics examine the independence of two
categorical vectors (here, a feature and the target vector):

$$\chi^2 = \sum_i (O_i - E_i) / E_i$$,

where $O_i$ is the observed count of the $i$-th combination (<feature_value>, <target_value>),
and $E_i$ is the expected count of that if they were independent.

For example, E_1 = P(<feature_value_1>) * P(<target_value_1>) =
= <count_of_feature_value_1> / <count_of_all_feature_values> * <count_of_target_value_1> / <count_of_all_target_values>

2. F-value scores examine if (when we group the numerical feature by the target vector)
the means for each group are significantly different.
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.feature_selection import chi2, f_classif, SelectKBest, SelectPercentile

iris = load_iris()
features = iris.data
features.shape
# array([[5.1, 3.5, 1.4, 0.2],
#        [4.9, 3. , 1.4, 0.2],
#        [4.7, 3.2, 1.3, 0.2],
#        [4.6, 3.1, 1.5, 0.2],
#        [5. , 3.6, 1.4, 0.2],
# ...
# (150, 4)
target = iris.target
np.unique(target)
# array([0, 1, 2])
target.shape
# (150,)

# # # 1. For categorical features, calculate a chi-square statistic between each feature and the target vector.

# Convert to categorical data by converting data to integers
features_categorical = features.astype(int)
# array([[5, 3, 1, 0],
#        [4, 3, 1, 0],
#        [4, 3, 1, 0],
#        [4, 3, 1, 0],
#        [5, 3, 1, 0],
# ...

# Select two features with highest chi-squared statistics.
# The parameter k determines the kept number of features.
chi2_selector = SelectKBest(chi2, k=2)
# SelectKBest(k=2, score_func=<function chi2 at 0x78063fe202c0>)
features_kbest = chi2_selector.fit_transform(features_categorical, target)
# array([[1, 0],
#        [1, 0],
#        [1, 0],
#        [1, 0],
#        [1, 0],
# ...

# Show results
print("Original number of features:", features_categorical.shape[1])
# Original number of features: 4
print("Reduced number of features:", features_kbest.shape[1])
# Reduced number of features: 2

# # # 2. For quantitative features, compute the ANOVA F-value between each feature and the target vector.

# Select two features with highest F-values
fvalue_selector = SelectKBest(f_classif, k=2)
# SelectKBest(k=2)
features_kbest = fvalue_selector.fit_transform(features, target)
# array([[1.4, 0.2],
#        [1.4, 0.2],
#        [1.3, 0.2],
#        [1.5, 0.2],
#        [1.4, 0.2],
# ...

print("Original number of features:", features.shape[1])
# Original number of features: 4
print("Reduced number of features:", features_kbest.shape[1])
# Reduced number of features: 2

# Use SelectPercentile to select the top n percent of features.
# Select top 75% of features with highest F-values.
fvalue_selector = SelectPercentile(f_classif, percentile=75)
# SelectPercentile(percentile=75)
features_kbest = fvalue_selector.fit_transform(features, target)
# array([[5.1, 1.4, 0.2],
#        [4.9, 1.4, 0.2],
#        [4.7, 1.3, 0.2],
#        [4.6, 1.5, 0.2],
#        [5. , 1.4, 0.2],
# ...

# Show results
print("Original number of features:", features.shape[1])
# Original number of features: 4
print("Reduced number of features:", features_kbest.shape[1])
# Reduced number of features: 3
