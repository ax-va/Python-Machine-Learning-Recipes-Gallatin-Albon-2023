"""
Describe the shape, data type, and format of a tensor along with the hardware it's using.
->
Inspect the shape, dtype, layout, and device attributes of the tensor.
"""
import torch

# Create a tensor
tensor = torch.tensor(
    [[1,2,3],
     [1,2,3]]
)
# tensor([[1, 2, 3],
#         [1, 2, 3]])

# Get the dimensions of the tensor
tensor.shape
# torch.Size([2, 3])
tensor.shape[0]
# 2
tensor.shape[1]
# 3

# Get the data type of items within the tensor
tensor.dtype
# torch.int64

# Get the memory layout (most common is "strided" used for dense tensors)
tensor.layout
# torch.strided

# Get the hardware the tensor is being stored on (CPU/GPU)
tensor.device
# device(type='cpu')
