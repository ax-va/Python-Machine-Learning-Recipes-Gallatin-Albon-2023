"""
Create a tensor.

See also:
PyTorch: Tensors
- https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html
"""
import torch

torch.__version__
# '2.3.1+cu121'

# Create a vector as a row
tensor_row = torch.tensor([1, 2, 3])
# tensor([1, 2, 3])

# Create a vector as a column
tensor_column = torch.tensor(
    [
        [1],
        [2],
        [3]
    ]
)
# tensor([[1],
#         [2],
#         [3]])

# Create a matrix tensor
torch.tensor(
    [[1, 2, 3],
     [4, 5, 6]]
)
# tensor([[1, 2, 3],
#         [4, 5, 6]])
