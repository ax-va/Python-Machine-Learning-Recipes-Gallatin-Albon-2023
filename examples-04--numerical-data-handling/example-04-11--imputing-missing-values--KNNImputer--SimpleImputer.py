"""
Impute (fill in, predict) missing values using Scikit-Learn's

- KNNImputer with k-nearest neighbors =>
problematic for big datasets because of computing distances between the missing value and every single observation =>
approximate nearest neighbors (ANN) more feasible.

- SimpleImputer with the feature’s mean, median, or most frequent value =>
often not as close to the true value as when we used KNN.

Simple imputing typically gives worse results than with KNN imputing, but appropriate for big datasets.
"""
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

# Make a simulated feature matrix
features, _ = make_blobs(
    n_samples=1000,
    n_features=2,
    random_state=1,
)
# array([[-3.05837272,  4.48825769],
#        [-8.60973869, -3.72714879],
#        [ 1.37129721,  5.23107449],
#        ...,
#        [-1.91854276,  4.59578307],
#        [-1.79600465,  4.28743568],
#        [-6.97684609, -8.89498834]])

# Standardize the features (to the normal distribution)
scaler = StandardScaler()
standardized_features = scaler.fit_transform(features)
# array([[ 0.87301861,  1.31426523],
#        [-0.67073178, -0.22369263],
#        [ 2.1048424 ,  1.45332359],
#        ...,
#        [ 1.18998798,  1.33439442],
#        [ 1.22406396,  1.27667052],
#        [-0.21664919, -1.19113343]])

# Replace the first feature's first value with a missing value
true_value = standardized_features[0, 0]
# 0.8730186113995938
standardized_features[0, 0] = np.nan
standardized_features
# array([[        nan,  1.31426523],
#        [-0.67073178, -0.22369263],
#        [ 2.1048424 ,  1.45332359],
#        ...,
#        [ 1.18998798,  1.33439442],
#        [ 1.22406396,  1.27667052],
#        [-0.21664919, -1.19113343]])

# # # KNNImputer

# Predict the missing values in the feature matrix
knn_imputer = KNNImputer(n_neighbors=5)  # Predict the missing value using the five closest observations
features_knn_imputed = knn_imputer.fit_transform(standardized_features)

# Compare true and imputed values
print("True Value:", true_value)
# True Value: 0.8730186113995938
print("Imputed Value:", features_knn_imputed[0, 0])
# Imputed Value: 1.0959262913919632

# # # SimpleImputer

# Create imputer using the "mean" strategy
mean_imputer = SimpleImputer(strategy="mean")

# Impute values
features_mean_imputed = mean_imputer.fit_transform(features)

# Compare true and imputed values
print("True Value:", true_value)
# True Value: 0.8730186113995938
print("Imputed Value:", features_mean_imputed[0, 0])
# Imputed Value: -3.058372724614996
