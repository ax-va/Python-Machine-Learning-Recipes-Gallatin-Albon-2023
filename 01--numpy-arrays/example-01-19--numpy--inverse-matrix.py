import numpy as np

matrix = np.array([[1, 4],
                   [2, 5]])
# array([[1, 4],
#        [2, 5]])

# Calculate inverse of matrix
np.linalg.inv(matrix)
# array([[-1.66666667,  1.33333333],
#        [ 0.66666667, -0.33333333]])

# Multiply matrix and its inverse
matrix @ np.linalg.inv(matrix)
# array([[1.00000000e+00, 0.00000000e+00],
#        [1.11022302e-16, 1.00000000e+00]])
