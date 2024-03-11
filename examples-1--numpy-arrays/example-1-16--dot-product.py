import numpy as np


vector_a = np.array([1, 2, 3])
# array([1, 2, 3])
vector_b = np.array([4, 5, 6])
# array([4, 5, 6])

# Calculate dot product
np.dot(vector_a, vector_b)
# 32

# In Python 3.5+, use @
vector_a @ vector_b
# 32
