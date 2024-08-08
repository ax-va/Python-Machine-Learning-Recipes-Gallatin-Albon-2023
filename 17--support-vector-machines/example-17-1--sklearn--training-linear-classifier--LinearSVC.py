"""
Use a *support vector classifier* (SVC) to find the line
in the two-dimensional space that maximizes the margins between two classes.
"""
import numpy as np
from sklearn.svm import LinearSVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

# Load data with only two classes and two features
iris = datasets.load_iris()
features = iris.data[:100, :2]
target = iris.target[:100]

# dimensions
features.shape
# (100, 2)
target.shape
# (100,)

# classes
np.unique(target)
# array([0, 1])

# Standardize features
features_standardized = StandardScaler().fit_transform(features)

# Create support vector classifier
svc = LinearSVC(
    # C is the penalty for misclassifying a data point:
    # smaller C -> underfitting (high bias but low variance, more misclassification of training data),
    # larger C -> overfitting (low bias but high variance, less misclassification of training data).
    C=1.0,
    dual="auto",
)
# Train model
model = svc.fit(features_standardized, target)

# Create new observation
new_observation = [[-2, 3]]
# Predict class of new observation
svc.predict(new_observation)
# array([0])

# Demonstrate that the data is linearly separable.
# Plot data points and color using their class.
colors = ["black" if c == 0 else "lightgrey" for c in target]
plt.scatter(features_standardized[:, 0], features_standardized[:, 1], c=colors)

# Create the line (hyperplane)
w = svc.coef_[0]
# array([ 1.68972244, -1.24087186])
a = -w[0] / w[1]
xx = np.linspace(-2.5, 2.5)
# array([-2.5       , -2.39795918, -2.29591837, -2.19387755, -2.09183673,
# ..
#         2.09183673,  2.19387755,  2.29591837,  2.39795918,  2.5       ])
xx.shape
# (50,)
svc.intercept_
# array([0.25980001])
yy = a * xx - (svc.intercept_[0]) / w[1]

# Plot the line (hyperplane)
plt.plot(xx, yy)
plt.axis("off")  # , plt.show();
plt.savefig('example-17-1--sklearn--training-linear-classifier--LinearSVC.svg')
plt.close()
