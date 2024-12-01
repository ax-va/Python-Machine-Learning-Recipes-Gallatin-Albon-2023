"""
Create PyTorch tensors from NumPy arrays.
->
Use PyTorch's from_numpy.

Notice:
PyTorch tensors and NumPy arrays can share the same memory to reduce overhead.

See also:
- PyTorch: Bridge with NumPy
https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#bridge-to-np-label
"""
import numpy as np
import torch

# Create a NumPy array
vector_row = np.array([1, 2, 3])

# Create a tensor from a NumPy array
tensor_row = torch.from_numpy(vector_row)
# tensor([1, 2, 3])
