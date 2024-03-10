import numpy as np

# Create 4x3 matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9],
                   [10, 11, 12]])
# array([[ 1,  2,  3],
#        [ 4,  5,  6],
#        [ 7,  8,  9],
#        [10, 11, 12]])

matrix.shape
# (4, 3)

matrix.size
# 12

# Reshape matrix into 2x6 matrix
matrix.reshape(2, 6)
# array([[ 1,  2,  3,  4,  5,  6],
#        [ 7,  8,  9, 10, 11, 12]])

matrix.reshape(2, 6).shape
# (2, 6)

matrix.reshape(2, 6).size
# 12

# -1 means "as many as needed"
matrix.reshape(1, -1)
# array([[ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12]])

matrix.reshape(12)
# array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])

# matrix.reshape(11)
# ValueError: cannot reshape array of size 12 into shape (11,)
