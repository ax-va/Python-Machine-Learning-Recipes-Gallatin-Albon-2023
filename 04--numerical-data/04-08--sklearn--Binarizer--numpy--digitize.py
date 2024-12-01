"""
Break a numerical feature up into discrete bins.

Two techniques:
- sklearn.preprocessing.Binarizer for a single threshold
- np.digitize also for multiple thesholds
"""
import numpy as np
from sklearn.preprocessing import Binarizer

ages = np.array([[6],
                 [12],
                 [20],
                 [36],
                 [38],
                 [45],
                 [51],
                 [65]])
# array([[ 6],
#        [12],
#        [20],
#        [36],
#        [38],
#        [45],
#        [51],
#        [65]])

# Create binarizer
binarizer = Binarizer(threshold=18)

# Transform feature
binarizer.fit_transform(ages)
# array([[0],
#        [0],
#        [1],
#        [1],
#        [1],
#        [1],
#        [1],
#        [1]])

# Break up numerical features according to multiple thresholds
np.digitize(ages, bins=[18, 21, 35, 65])  # The left edge is default: right=False
# array([[0],
#        [0],
#        [1],
#        [3],
#        [3],
#        [3],
#        [3],
#        [4]])

# The left edge means:
# 6, 12 < 18,
# 18 <= 20 < 21,
# nothing in [21, 35),
# 35 <= 36, 38, 45, 51 < 65,
# 65 <= 65

# Switch to the right edge
np.digitize(ages, bins=[18, 21, 35, 65], right=True)
# array([[0],
#        [0],
#        [1],
#        [3],
#        [3],
#        [3],
#        [3],
#        [3]])

# Use digitize to binarize features like Binarizer by specifying only a single threshold
np.digitize(ages, bins=[18])
# array([[0],
#        [0],
#        [1],
#        [1],
#        [1],
#        [1],
#        [1],
#        [1]])
