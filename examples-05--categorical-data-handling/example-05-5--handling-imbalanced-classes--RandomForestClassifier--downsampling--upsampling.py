"""
Handle the class imbalance with:
1. Scikit-Learn classifiers have a "class_weight" parameter, also RandomForestClassifier considered below
2. downsampling the majority class (the class with more observations)
3. upsampling the minority class (the class with fewer observations)
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

iris = load_iris()

# Create feature matrix
features = iris.data
# array([[5.1, 3.5, 1.4, 0.2],
#        [4.9, 3. , 1.4, 0.2],
#        [4.7, 3.2, 1.3, 0.2],
# ...
#        [6.5, 3. , 5.2, 2. ],
#        [6.2, 3.4, 5.4, 2.3],
#        [5.9, 3. , 5.1, 1.8]])

# Create target vector
target = iris.target
# array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

target.shape
# (110,)

# Remove first 40 observations
features = features[40:, :]
target = target[40:]

# Create binary target vector indicating if class 0 that is the imbalanced target vector
target = np.where((target == 0), 0, 1)
# array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

# # # 1. RandomForestClassifier

# Create weights for RandomForestClassifier
weights = {0: 0.1, 1: 0.9}

RandomForestClassifier(class_weight=weights)
# RandomForestClassifier(class_weight={0: 0.1, 1: 0.9})

# Automatically create weights inversely proportional to class frequencies
RandomForestClassifier(class_weight="balanced")
# RandomForestClassifier(class_weight='balanced')

# The example has been not completed ...

# # # 2-3. downsampling and upsampling

# indices of each class's observations
indices_class_0 = np.where(target == 0)[0]
# array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
indices_class_1 = np.where(target == 1)[0]
# array([ 10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,
#         23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,
#         36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,
#         49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,
#         62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,
#         75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,
#         88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100,
#        101, 102, 103, 104, 105, 106, 107, 108, 109])

# number of observations in each class
num_class_0 = len(indices_class_0)
# 10
num_class_1 = len(indices_class_1)
# 100

# 2. Downsample the majority class (the class with more observations)

# For every observation of class 0, randomly sample from class 1 without repeating values
indices_class_1_downsampled = np.random.choice(num_class_1, size=num_class_0, replace=False)
# array([76, 44, 52, 70, 24, 73, 87, 74, 53, 72])

# Join together class 0's target vector with the downsampled class 1's target vector
np.hstack(
    (
        target[indices_class_0],
        target[indices_class_1_downsampled],
    )
)
# array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

# Join together class 0's feature matrix with the downsampled class 1's feature matrix
np.vstack(
    (
        features[indices_class_0, :],
        features[indices_class_1_downsampled, :],
    )
)[0:5]
# array([[5. , 3.5, 1.3, 0.3],
#        [4.5, 2.3, 1.3, 0.3],
#        [4.4, 3.2, 1.3, 0.2],
#        [5. , 3.5, 1.6, 0.6],
#        [5.1, 3.8, 1.9, 0.4]])

# 3. Upsampling is similarly to downsampling, but in reverse

# For every observation in class 1, randomly sample from class 0 with repeating values
indices_class_0_upsampled = np.random.choice(indices_class_0, size=num_class_1, replace=True)
# array([7, 5, 0, 9, 9, 4, 3, 9, 6, 7, 4, 1, 2, 4, 6, 1, 3, 2, 0, 7, 5, 3,
#        5, 3, 5, 8, 2, 2, 7, 4, 6, 7, 3, 8, 7, 8, 4, 5, 6, 9, 1, 3, 7, 2,
#        0, 2, 0, 7, 8, 7, 8, 4, 1, 7, 6, 6, 2, 6, 7, 4, 8, 2, 1, 2, 3, 2,
#        2, 9, 0, 3, 4, 8, 5, 1, 1, 7, 9, 4, 0, 1, 2, 9, 2, 9, 4, 1, 0, 6,
#        5, 2, 4, 3, 2, 7, 9, 4, 3, 6, 0, 8])

# Join together class 0's upsampled target vector with class 1's target vector
np.concatenate(
    (
        target[indices_class_0_upsampled],
        target[indices_class_1]
    )
)
# array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1])

# Note: here, np.hstack and np.concatenate are equivalent.

# Join together class 0's upsampled feature matrix with class 1's feature matrix
np.vstack(
    (
        features[indices_class_0_upsampled, :],
        features[indices_class_1, :]
    )
)[0:5]
# array([[4.6, 3.2, 1.4, 0.2],
#        [4.8, 3. , 1.4, 0.3],
#        [5. , 3.5, 1.3, 0.3],
#        [5. , 3.3, 1.4, 0.2],
#        [5. , 3.3, 1.4, 0.2]])
