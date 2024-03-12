import numpy as np

# Set seed
np.random.seed(0)

# Generate three random floats between 0.0 and 1.0
np.random.random(3)
# array([0.5488135 , 0.71518937, 0.60276338])

# Generate three random integers between 0 and 10
np.random.randint(0, 11, 3)
# array([3, 7, 9])

# Draw three numbers from a normal distribution with mean=0.0 and std=1.0
np.random.normal(0.0, 1.0, 3)
# array([-1.42232584,  1.52006949, -0.29139398])

# Draw three numbers from a logistic distribution with mean=0.0 and scale=1.0
np.random.logistic(0.0, 1.0, 3)
# array([-0.98118713, -0.08939902,  1.46416405])

# Draw three numbers greater than or equal to 1.0 and less than 2.0
np.random.uniform(1.0, 2.0, 3)
# array([1.47997717, 1.3927848 , 1.83607876])
