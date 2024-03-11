import numpy as np


matrix = np.array([[1, 2, 3],
                   [2, 4, 6],
                   [3, 8, 9]])
# array([[1, 2, 3],
#        [2, 4, 6],
#        [3, 8, 9]])

matrix.diagonal()
# array([1, 4, 9])

# Return diagonal one above the main diagonal
matrix.diagonal(offset=1)
# array([2, 6])

# Return diagonal one below the main diagonal
matrix.diagonal(offset=-1)
# array([2, 8])
