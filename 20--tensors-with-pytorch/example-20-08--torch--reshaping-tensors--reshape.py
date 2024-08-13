"""
Change the shape (number of rows and columns) of a tensor without changing the element values.
->
Use the PyTorch's reshape method.
"""
import torch

# Create 4x3 tensor
tensor = torch.tensor(
    [[ 1,  2,  3],
     [ 4,  5,  6],
     [ 7,  8,  9],
     [10, 11, 12]]
)
# tensor([[ 1,  2,  3],
#         [ 4,  5,  6],
#         [ 7,  8,  9],
#         [10, 11, 12]])

tensor.shape
# torch.Size([4, 3])

# Reshape tensor into 2 x 6 tensor
tensor.reshape(2, 6)
# tensor([[ 1,  2,  3,  4,  5,  6],
#         [ 7,  8,  9, 10, 11, 12]])

tensor.shape
# torch.Size([4, 3])

tensor.reshape(2, 6).shape
# torch.Size([2, 6])
