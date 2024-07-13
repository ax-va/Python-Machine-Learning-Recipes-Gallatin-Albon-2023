import numpy as np

matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
# array([[1, 2, 3],
#        [4, 5, 6],
#        [7, 8, 9]])

# Flatten matrix
matrix.flatten()
# array([1, 2, 3, 4, 5, 6, 7, 8, 9])

matrix
# array([[1, 2, 3],
#        [4, 5, 6],
#        [7, 8, 9]])

# Alternatively, use reshape
matrix.reshape(1, -1)
# array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])

matrix
# array([[1, 2, 3],
#        [4, 5, 6],
#        [7, 8, 9]])

# Alternatively to flatten, use ravel
matrix.ravel()
# array([1, 2, 3, 4, 5, 6, 7, 8, 9])

matrix
# array([[1, 2, 3],
#        [4, 5, 6],
#        [7, 8, 9]])

# 1) The primary functional difference is thatflatten is a method of an ndarray object
# and hence can only be called for true numpy arrays. In contrast ravel() is a
# library-level function and hence can be called on any object that can successfully
# be parsed. For example ravel() will work on a list of ndarrays, while flatten won't.
# https://stackoverflow.com/a/28846614

# 2) ravel can be faster than flatten

matrix_a = np.array([[1, 2],
                     [3, 4]])
# array([[1, 2],
#        [3, 4]])

matrix_b = np.array([[5, 6],
                     [7, 8]])
# array([[5, 6],
#        [7, 8]])

# Create a list of matrices
matrix_list = [matrix_a, matrix_b]
# [array([[1, 2],
#         [3, 4]]),
#  array([[5, 6],
#         [7, 8]])]

# Flatten the entire list of matrices
np.ravel(matrix_list)
# array([1, 2, 3, 4, 5, 6, 7, 8])
