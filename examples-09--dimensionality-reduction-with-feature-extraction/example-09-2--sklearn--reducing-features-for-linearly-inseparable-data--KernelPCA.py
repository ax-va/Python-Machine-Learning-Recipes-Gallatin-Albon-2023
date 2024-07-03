"""
Reduce the dimensions for linearly inseparable data.
->
Kernels can project the linearly inseparable data into a higher dimension
to make it linearly separable.  This is called the "kernel trick".

A common kernel to use is the Gaussian radial basis function kernel (rbf).
There are also the polynomial kernel (poly) and sigmoid kernel (sigmoid).

One downside of kernel PCA is specifying a number of components, no variance,
and its own hyperparameters like gamma for RBF.

See also:
Scikit-Learn: KernelPCA
https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html

Sebastian Raschka: Kernel tricks and nonlinear dimensionality reduction via RBF kernel PCA
https://sebastianraschka.com/Articles/2014_kernel_pca.html
"""
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_circles
from matplotlib import pyplot as plt

# Create linearly inseparable data
features, target = make_circles(n_samples=1000, random_state=1, noise=0.1, factor=0.1)
# array([[ 0.23058395, -0.10671314],
#        [-0.0834218 , -0.22647078],
#        [ 0.9246533 , -0.71492522],
#        ...,
#        [ 0.02517206,  0.00964548],
#        [-0.92836187,  0.06693357],
#        [ 1.03502248,  0.54878286]])

plt.scatter(features[:, 0][target == 0], features[:, 1][target == 0])
plt.scatter(features[:, 0][target == 1], features[:, 1][target == 1])
# plt.show()
plt.savefig(f'example-09-2--sklearn--reducing-features-for-linearly-inseparable-data--KernelPCA.svg')
plt.close()

# Apply kernel PCA with radius basis function (RBF) kernel
kpca = KernelPCA(kernel="rbf", gamma=15, n_components=1)
features_kpca = kpca.fit_transform(features)

# Print results
print("Original number of features:", features.shape[1])
# Original number of features: 2
print("Reduced number of features:", features_kpca.shape[1])
# Reduced number of features: 1
