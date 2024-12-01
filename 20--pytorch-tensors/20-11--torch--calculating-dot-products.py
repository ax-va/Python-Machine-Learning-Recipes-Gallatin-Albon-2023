"""
Calculate the dot product of two tensors.
->
Use the dot method.

See also:
- Vectorization and Broadcasting with PyTorch
https://blog.paperspace.com/pytorch-vectorization-and-broadcasting/
"""
import torch

# Create one tensor
tensor_1 = torch.tensor([1, 2, 3])
# tensor([1, 2, 3])

# Create another tensor
tensor_2 = torch.tensor([4, 5, 6])
# tensor([4, 5, 6])

# Calculate the dot product of the two tensors
tensor_1.dot(tensor_2)
# tensor(32)
