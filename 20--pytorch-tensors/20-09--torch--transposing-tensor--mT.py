"""
Transpose a tensor.
->
Use the mT method.

Notice:
The T method used for NumPy arrays is supported
in PyTorch only with tensors of two dimensions.
"""
import torch

# Create a two-dimensional tensor
tensor = torch.tensor([[[1, 2, 3]]])
# tensor([[[1, 2, 3]]])

# Transpose tensor
tensor.mT
# tensor([[[1],
#          [2],
#          [3]]])

tensor.ndim
# 3

# tensor.shape
torch.Size([1, 1, 3])

# Another method to transpose is to use the permute method
tensor.permute(*torch.arange(tensor.ndim - 1, -1, -1))
# tensor([[[1]],
#
#         [[2]],
#
#         [[3]]])


# The permute method can be also applied to one-dimensional tensors, while mT cannot
tensor_1dim = torch.tensor([1, 2, 3])
# tensor([1, 2, 3])

# tensor_1dim.mT
# # RuntimeError: tensor.mT is only supported on matrices or batches of matrices. Got 1-D tensor.

tensor_1dim.permute(*torch.arange(tensor_1dim.ndim - 1, -1, -1))
# tensor([1, 2, 3])
