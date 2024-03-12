import numpy as np

matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
# array([[1, 2, 3],
#        [4, 5, 6],
#        [7, 8, 9]])

# Transpose matrix
matrix.T
# array([[1, 4, 7],
#        [2, 5, 8],
#        [3, 6, 9]])

# Vector as dim=1-array will be not transposed
np.array([1, 2, 3, 4, 5, 6]).T
# array([1, 2, 3, 4, 5, 6])

# Transpose row vector
np.array([[1, 2, 3, 4, 5, 6]]).T
# array([[1],
#        [2],
#        [3],
#        [4],
#        [5],
#        [6]])
