import numpy as np

# Create row vector
vector = np.array([1, 2, 3, 4, 5, 6])
# array([1, 2, 3, 4, 5, 6])

# Select third element of vector
vector[2]
# 3

# Select all elements of a vector
vector[:]
# array([1, 2, 3, 4, 5, 6])

# Select everything up to and including the third element
vector[:3]
# array([1, 2, 3])

# Select everything after the third element
vector[3:]
# array([4, 5, 6])

# Select the last element
vector[-1]
# 6

# Reverse the vector
vector[::-1]
# array([6, 5, 4, 3, 2, 1])

# Create matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
# array([[1, 2, 3],
#        [4, 5, 6],
#        [7, 8, 9]])

# Select second row, second column
matrix[1, 1]
# 5

# Select the first two rows and all columns of a matrix
matrix[:2, :]
# array([[1, 2, 3],
#        [4, 5, 6]])

# Select all rows and the second column
matrix[:, 1:2]
# array([[2],
#        [5],
#        [8]])
