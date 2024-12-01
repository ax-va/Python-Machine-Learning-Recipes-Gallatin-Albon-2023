import numpy as np

matrix_a = np.array([[1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 2]])
# array([[1, 1, 1],
#        [1, 1, 1],
#        [1, 1, 2]])

matrix_b = np.array([[1, 3, 1],
                     [1, 3, 1],
                     [1, 3, 8]])
# array([[1, 3, 1],
#        [1, 3, 1],
#        [1, 3, 8]])

# Add two matrices
np.add(matrix_a, matrix_b)
# array([[ 2,  4,  2],
#        [ 2,  4,  2],
#        [ 2,  4, 10]])

matrix_a + matrix_b
# array([[ 2,  4,  2],
#        [ 2,  4,  2],
#        [ 2,  4, 10]])

# Subtract two matrices
np.subtract(matrix_a, matrix_b)
# array([[ 0, -2,  0],
#        [ 0, -2,  0],
#        [ 0, -2, -6]])

matrix_a - matrix_b
# array([[ 0, -2,  0],
#        [ 0, -2,  0],
#        [ 0, -2, -6]])
