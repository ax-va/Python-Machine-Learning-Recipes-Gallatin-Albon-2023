"""
Transform a tensor into one dimension.
->
Use the PyTorch's flatten method.
"""
import torch

# Create tensor
tensor = torch.tensor(
    [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]
)
# tensor([[1, 2, 3],
#         [4, 5, 6],
#         [7, 8, 9]])

# Flatten tensor
tensor.flatten()
# tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
