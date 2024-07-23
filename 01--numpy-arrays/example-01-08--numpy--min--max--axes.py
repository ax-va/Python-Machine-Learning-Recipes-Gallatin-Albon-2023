import numpy as np

matrix = np.array([[1, 2],
                   [4, 5],
                   [7, 8]])
# array([[1, 2],
#        [4, 5],
#        [7, 8]])

np.min(matrix)
# 1

np.max(matrix)
# 8

# Find maximum element in each column, i.e. along axis 0 for each element fixed in axis 1
np.max(matrix, axis=0)
# array([7, 8])

# Find maximum element in each row, i.e. along axis 1 for each element fixed in axis 0
np.max(matrix, axis=1)
# array([2, 5, 8])

"""
|-------> axis 1
|  1  2
|  4  5
|  7  8
V
axis 0
"""
