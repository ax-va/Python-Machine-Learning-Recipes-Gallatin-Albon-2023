import numpy as np

vector_a = np.array([1, 2, 3])
# array([1, 2, 3])
vector_b = np.array([4, 5, 6])
# array([4, 5, 6])

# Calculate dot product
np.dot(vector_a, vector_b)
# 32

# In Python 3.5+, use @
vector_a @ vector_b
# 32

matrix_a = np.array([[1, 1, 1],
                     [1, 2, 3]])
# array([[1, 1, 1],
#        [1, 2, 3]])

matrix_b = np.array([[1, 1],
                     [1, 2],
                     [1, 3]])
# array([[1, 1],
#        [1, 2],
#        [1, 3]])

np.dot(matrix_a, matrix_b)
# array([[ 3,  6],
#        [ 6, 14]])

matrix_a @ matrix_b
# array([[ 3,  6],
#        [ 6, 14]])
