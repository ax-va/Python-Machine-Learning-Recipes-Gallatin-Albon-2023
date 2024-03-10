import numpy as np

# Create matrix
matrix = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])
# array([[ 1,  2,  3,  4],
#        [ 5,  6,  7,  8],
#        [ 9, 10, 11, 12]])

# Get number of rows and columns
matrix.shape
# (3, 4)

# 3 rows and 4 columns, i.e. dim=3 along axis=0 and dim=4 along axis=1

# Get number of elements (rows * columns)
matrix.size
# 12

# Get number of dimensions
matrix.ndim
# 2
