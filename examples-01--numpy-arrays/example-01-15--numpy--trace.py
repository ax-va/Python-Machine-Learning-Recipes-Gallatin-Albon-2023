import numpy as np


matrix = np.array([[1, 2, 3],
                   [2, 4, 6],
                   [3, 8, 9]])
# array([[1, 2, 3],
#        [2, 4, 6],
#        [3, 8, 9]])

# Return trace
matrix.trace()
# 14

# Alternatively, return the diagonal of a matrix and calculate its sum
sum(matrix.diagonal())
# 14
