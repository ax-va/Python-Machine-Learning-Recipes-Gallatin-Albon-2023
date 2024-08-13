"""
Find the minimum or maximum value in a tensor.
->
Use the PyTorch's min and max methods.
"""
import torch

# Create a tensor
tensor = torch.tensor([1, 2, 3])

# Find the maximum
tensor.max()
# tensor(3)

# Find the minimum
tensor.min()
# tensor(1)

# Create a multidimensional tensor
tensor_matrix = torch.tensor(
    [[1, 2, 3],
     [3, 4, 5]]
)
# tensor([[1, 2, 3],
#         [3, 4, 5]])

tensor_matrix.min()
# tensor(1)

tensor_matrix.max()
# tensor(5)

# Find the minimum in each column, i.e. going over axis 0
tensor_matrix.min(dim=0)
# torch.return_types.min(
# values=tensor([1, 2, 3]),
# indices=tensor([0, 0, 0]))

# Find the minimum in each row, i.e. going over axis 1
tensor_matrix.min(dim=1)
# torch.return_types.min(
# values=tensor([1, 3]),
# indices=tensor([0, 0]))

"""
|-------> axis 1
|  1  2
|  4  5
|  7  8
V
axis 0
"""