import numpy as np

# recommended
matrix = np.array([[1, 2],
                   [3, 4],
                   [5, 6]])
# array([[1, 2],
#        [3, 4],
#        [5, 6]])

# not recommended
matrix_object = np.mat([[1, 2],
                        [1, 2],
                        [1, 2]])
# matrix([[1, 2],
#         [1, 2],
#         [1, 2]])
