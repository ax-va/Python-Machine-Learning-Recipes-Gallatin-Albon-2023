"""
Replace missing categorical feature values with predicted values by
- the k-nearest neighbors (KNN) classifier => KNeighborsClassifier
- filling-in missing values with the feature's most frequent value => SimpleImputer
"""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer

# Create feature matrix with categorical feature: the first row contains categorical values (classes)
X = np.array([[0, 2.10, 1.45],
              [1, 1.18, 1.33],
              [0, 1.22, 1.27],
              [1, -0.21, -1.19]])
# array([[ 0.  ,  2.1 ,  1.45],
#        [ 1.  ,  1.18,  1.33],
#        [ 0.  ,  1.22,  1.27],
#        [ 1.  , -0.21, -1.19]])

# Create feature matrix with missing values in the categorical feature
X_with_nan = np.array([[np.nan, 0.87, 1.31],
                       [np.nan, -0.67, -0.22]])
# array([[  nan,  0.87,  1.31],
#        [  nan, -0.67, -0.22]])

# 1. KNeighborsClassifier

# Train KNN learner
clf = KNeighborsClassifier(n_neighbors=3, weights='distance')
trained_model = clf.fit(X=X[:, 1:], y=X[:, 0])

# Predict class of missing values
imputed_values = trained_model.predict(X_with_nan[:, 1:])
# array([0., 1.])

imputed_values.reshape(-1, 1)
# array([[0.],
#        [1.]])

# Join column of predicted class with their other features
X_with_imputed = np.hstack((imputed_values.reshape(-1, 1), X_with_nan[:, 1:]))
# array([[ 0.  ,  0.87,  1.31],
#        [ 1.  , -0.67, -0.22]])

# Join two feature matrices
np.vstack((X_with_imputed, X))
# array([[ 0.  ,  0.87,  1.31],
#        [ 1.  , -0.67, -0.22],
#        [ 0.  ,  2.1 ,  1.45],
#        [ 1.  ,  1.18,  1.33],
#        [ 0.  ,  1.22,  1.27],
#        [ 1.  , -0.21, -1.19]])

# 2. SimpleImputer

# Alternatively, fill in missing values with the feature's most frequent value

# Join the two feature matrices
X_complete = np.vstack((X_with_nan, X))
# array([[  nan,  0.87,  1.31],
#        [  nan, -0.67, -0.22],
#        [ 0.  ,  2.1 ,  1.45],
#        [ 1.  ,  1.18,  1.33],
#        [ 0.  ,  1.22,  1.27],
#        [ 1.  , -0.21, -1.19]])

imputer = SimpleImputer(strategy='most_frequent')
imputer.fit_transform(X_complete)
# array([[ 0.  ,  0.87,  1.31],
#        [ 0.  , -0.67, -0.22],
#        [ 0.  ,  2.1 ,  1.45],
#        [ 1.  ,  1.18,  1.33],
#        [ 0.  ,  1.22,  1.27],
#        [ 1.  , -0.21, -1.19]])
