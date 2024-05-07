import numpy as np
from scipy import sparse

matrix = np.array([[0, 0],
                   [0, 1],
                   [3, 0]])
# array([[0, 0],
#        [0, 1],
#        [3, 0]])

# Create compressed sparse row (CSR) matrix
matrix_sparse = sparse.csr_matrix(matrix)
# <3x2 sparse matrix of type '<class 'numpy.int64'>'
#         with 2 stored elements in Compressed Sparse Row format>
print(matrix_sparse)
#   (1, 1)        1
#   (2, 0)        3

# Create larger matrix
matrix_large = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                         [3, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
# array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#        [3, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

# Create compressed sparse row (CSR) matrix
matrix_large_sparse = sparse.csr_matrix(matrix_large)
# <3x10 sparse matrix of type '<class 'numpy.int64'>'
#         with 2 stored elements in Compressed Sparse Row format>

# View original sparse matrix
print(matrix_large_sparse)
#   (1, 1)        1
#   (2, 0)        3

matrix_large_sparse.toarray()
# array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#        [3, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
