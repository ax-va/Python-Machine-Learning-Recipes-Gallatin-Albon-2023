import numpy as np

# Create matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
# array([[1, 2, 3],
#        [4, 5, 6],
#        [7, 8, 9]])

# mean
np.mean(matrix)
# 5.0

# variance
np.var(matrix)
# 6.666666666666667

# standard deviation
np.std(matrix)
# 2.581988897471611

# Find the mean value in each column
np.mean(matrix, axis=0)
# array([4., 5., 6.])
