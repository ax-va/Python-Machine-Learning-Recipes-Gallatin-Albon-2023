import numpy as np

matrix = np.array([[1, 1, 1],
                   [1, 1, 10],
                   [1, 1, 15]])
# array([[ 1,  1,  1],
#        [ 1,  1, 10],
#        [ 1,  1, 15]])

# Return matrix rank
np.linalg.matrix_rank(matrix)
# 2
