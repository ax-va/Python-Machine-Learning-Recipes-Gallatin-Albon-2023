"""
Train a support vector classifier, when classes are linearly inseparable.
->
Train an extension of a support vector machine using
kernel functions to create nonlinear decision boundaries.

A support vector classifier can be represented as
$$
f(x_i) = \beta_0 + \sum_{s \n S} \alpha_i K(x_s, x_i)
$$,
where
$\beta_0$ is the bias,
$S$ is the set of all support vector observations,
$\alpha$ is the model parameter to be learned,
$x_s$ and $x_i$ are two support vector observations, and
the kernel function $K$ support vector observations $x_s$ and $x_i$.

K determines the type of hyperplane used to separate classes:

- linear kernel for a basic linear hyperplane
$$
K(x_s, x_i) = \sum_{j=1}^p x_{js} x_{ji}
$$,
where $p$ is the number of features;

- polynomial kernel for a nonlinear decision boundary
$$
K(x_s, x_i) = (r + \gamma \sum_{j=1}^p x_{js} x_{ji})^d
$$;

- radial basis function kernel (one of the most common kernels)
$$
K(x_s, x_i) = \exp{-\gamma \sum_{j=1}^p (x_{js} x_{ji})^2}
$$,
where $\gamme > 0$.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC

# Set randomization seed
np.random.seed(0)
# Generate two features
features = np.random.randn(200, 2)
# array([[ 1.76405235,  0.40015721],
#        [ 0.97873798,  2.2408932 ],
#        [ 1.86755799, -0.97727788],
# ...
#        [-0.29183736, -0.76149221],
#        [ 0.85792392,  1.14110187],
#        [ 1.46657872,  0.85255194]])
# Generate linearly inseparable classes
# with two regions for class 0 and two regions for class 1.
target_xor = np.logical_xor(features[:, 0] > 0, features[:, 1] > 0)
target = np.where(target_xor, 0, 1)
# array([1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0,
# ...
#        0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1,
#        1, 1])

# Create a support vector machine with a radial basis function kernel
svc = SVC(
    kernel="rbf",  # radial basis function kernel
    random_state=0,
    gamma=1,  # parameter $\gamma$ in the kernel
    # C is the penalty for misclassifying a data point:
    # smaller C -> underfitting (high bias but low variance, more misclassification of training data),
    # larger C -> overfitting (low bias but high variance, less misclassification of training data).
    C=1,
)
# Train the classifier
model = svc.fit(features, target)

# Two ways to predict
model.predict([[0.75, .75]])
# array([1])
svc.predict([[0.75, .75]])
# array([1])

# Train again with inverse target classes
svc.fit(features, np.where(target, 0, 1))
# Changes in classifier change "model" too
model.predict([[0.75, .75]])
# array([0])
svc.predict([[0.75, .75]])
# array([0])

# # Plot observations and decision boundary hyperplane


def plot_decision_regions(X, y, classifier):
    cmap = ListedColormap(("red", "blue"))
    xx1, xx2 = np.meshgrid(
        np.arange(-3, 3, 0.02),
        np.arange(-3, 3, 0.02),
    )
    # [array([[-3.  , -2.98, -2.96, ...,  2.94,  2.96,  2.98],
    #         [-3.  , -2.98, -2.96, ...,  2.94,  2.96,  2.98],
    #         [-3.  , -2.98, -2.96, ...,  2.94,  2.96,  2.98],
    #         ...,
    #         [-3.  , -2.98, -2.96, ...,  2.94,  2.96,  2.98],
    #         [-3.  , -2.98, -2.96, ...,  2.94,  2.96,  2.98],
    #         [-3.  , -2.98, -2.96, ...,  2.94,  2.96,  2.98]]),
    #  array([[-3.  , -3.  , -3.  , ..., -3.  , -3.  , -3.  ],
    #         [-2.98, -2.98, -2.98, ..., -2.98, -2.98, -2.98],
    #         [-2.96, -2.96, -2.96, ..., -2.96, -2.96, -2.96],
    #         ...,
    #         [ 2.94,  2.94,  2.94, ...,  2.94,  2.94,  2.94],
    #         [ 2.96,  2.96,  2.96, ...,  2.96,  2.96,  2.96],
    #         [ 2.98,  2.98,  2.98, ...,  2.98,  2.98,  2.98]])]
    # xx1.shape
    # (300, 300)
    # xx2.shape
    # (300, 300)
    Z = classifier.predict(
        np.array(
            [xx1.ravel(), xx2.ravel()]
        ).T
        # array([[-3.  , -3.  ],
        #        [-2.98, -3.  ],
        #        [-2.96, -3.  ],
        #        ...,
        #        [ 2.94,  2.98],
        #        [ 2.96,  2.98],
        #        [ 2.98,  2.98]])
    )
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.1, cmap=cmap)
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            x=X[y == cl, 0], y=X[y == cl, 1],
            alpha=0.8,
            color=cmap(idx),
            marker="+",
            label=cl,
        )


# Create support vector classifier with a linear kernel
svc_linear = SVC(kernel="linear", random_state=0, C=1)
# Train model
svc_linear.fit(features, target)
# Plot observations and hyperplane
plot_decision_regions(features, target, classifier=svc_linear)
plt.axis("off")  # , plt.show();
plt.savefig('example-17-2-1--sklearn--handling-linearly-inseparable-classes-using-kernels--SVC--kernel.svg')
plt.close()

# Create a support vector machine with a radial basis function kernel
svc = SVC(kernel="rbf", random_state=0, gamma=1, C=1)
# Train the classifier
model = svc.fit(features, target)
# Plot observations and hyperplane
plot_decision_regions(features, target, classifier=svc)
plt.axis("off")  # , plt.show();
plt.savefig('example-17-2-2--sklearn--handling-linearly-inseparable-classes-using-kernels--SVC--kernel.svg')
plt.close()
