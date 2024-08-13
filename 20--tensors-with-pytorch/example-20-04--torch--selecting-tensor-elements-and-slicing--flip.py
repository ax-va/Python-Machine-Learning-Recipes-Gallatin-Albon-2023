"""
Select specific elements of a tensor.
->
Use NumPy-like indexing and slicing to return elements.

Difference:
PyTorch tensors do not yet support negative steps when slicing.

PyTorch: Operations on Tensors
https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html#operations-on-tensors
"""
import torch

# Create vector tensor
vector = torch.tensor([1, 2, 3, 4, 5, 6])
# tensor([1, 2, 3, 4, 5, 6])

# Create matrix tensor
matrix = torch.tensor(
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
)
# tensor([[1, 2, 3],
#         [4, 5, 6],
#         [7, 8, 9]])

# Select third element of vector.
# Tensor is return instead of the value of the object itself.
vector[2]
# tensor(3)

# Select second row, second column.
# Tensor is return instead of the value of the object itself.
matrix[1, 1]
# tensor(5)

# Slicing: select all elements of a vector
vector[:]
# tensor([1, 2, 3, 4, 5, 6])

# Select everything up to and including the third element
vector[:3]
# tensor([1, 2, 3])

# Select everything after the third element
vector[3:]
# tensor([4, 5, 6])

# Select the last element.
# Tensor is return instead of the value of the object itself.
vector[-1]
# tensor(6)

# Select the first two rows and all columns of a matrix
matrix[:2, :]
# tensor([[1, 2, 3],
#         [4, 5, 6]])

# Select all rows and the second column
matrix[:, 1:2]
# tensor([[2],
#         [5],
#         [8]])

# # Difference:
# # PyTorch tensors do not yet support negative steps when slicing.
# vector[::-1]
# # ValueError: step must be greater than zero

# Instead, use the "flip" method to reverse a tensor
vector.flip(dims=(-1, ))
# tensor([6, 5, 4, 3, 2, 1])
