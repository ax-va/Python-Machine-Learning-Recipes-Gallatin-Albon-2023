"""
Rescale the matrix of observations to have unit norm (||row|| = 1 for all rows).
"""
import numpy as np
from sklearn.preprocessing import Normalizer

# Create feature matrix: each row is an observation of the features x_1 and x_2
matrix = np.array([[0.5, 0.5],
                   [1.1, 3.4],
                   [1.5, 20.2],
                   [1.63, 34.4],
                   [10.9, 3.3]])
# array([[ 0.5 ,  0.5 ],
#        [ 1.1 ,  3.4 ],
#        [ 1.5 , 20.2 ],
#        [ 1.63, 34.4 ],
#        [10.9 ,  3.3 ]])

# Create normalizer
normalizer = Normalizer(norm="l2")  # l2 = Euclidean norm

# Transform matrix
normalizer.transform(matrix)
# array([[0.70710678, 0.70710678],
#        [0.30782029, 0.95144452],
#        [0.07405353, 0.99725427],
#        [0.04733062, 0.99887928],
#        [0.95709822, 0.28976368]])

matrix_l2_norm = Normalizer(norm="l2").transform(matrix)
# array([[0.70710678, 0.70710678],
#        [0.30782029, 0.95144452],
#        [0.07405353, 0.99725427],
#        [0.04733062, 0.99887928],
#        [0.95709822, 0.28976368]])

# Check L2 norms
np.sqrt(matrix_l2_norm[:, 0] ** 2 + matrix_l2_norm[:, 1] ** 2)
# array([1., 1., 1., 1., 1.])

matrix_l1_norm = Normalizer(norm="l1").transform(matrix)  # l1 = Manhattan norm = Taxicab norm
# array([[0.5       , 0.5       ],
#        [0.24444444, 0.75555556],
#        [0.06912442, 0.93087558],
#        [0.04524008, 0.95475992],
#        [0.76760563, 0.23239437]])

# Check L1 norms
matrix_l1_norm[:, 0] + matrix_l1_norm[:, 1]
# array([1., 1., 1., 1., 1.])
