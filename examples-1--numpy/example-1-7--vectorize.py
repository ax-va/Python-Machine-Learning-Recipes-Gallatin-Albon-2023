import numpy as np

# Create matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
# array([[1, 2, 3],
#        [4, 5, 6],
#        [7, 8, 9]])

# Create function that adds 100 to something
add_100 = lambda i: i + 100

# Create vectorized function
vectorized_add_100 = np.vectorize(add_100)

# Apply function to all elements in matrix
vectorized_add_100(matrix)
# array([[101, 102, 103],
#        [104, 105, 106],
#        [107, 108, 109]])

# vectorize is essentially a for loop over the elements and does not increase performance.

# Here, a better solution is broadcasting.

# Add 100 to all elements
matrix + 100
# array([[101, 102, 103],
#        [104, 105, 106],
#        [107, 108, 109]])
