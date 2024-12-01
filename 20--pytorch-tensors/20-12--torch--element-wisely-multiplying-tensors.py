"""
Multiply two tensors.
->
Use basic Python arithmetic operators.
"""
import torch

# Create one tensor
tensor_1 = torch.tensor([1, 2, 3])
# tensor([1, 2, 3])

# Create another tensor
tensor_2 = torch.tensor([4, 5, 6])
# tensor([4, 5, 6])

# Multiply the two tensors
tensor_1 * tensor_2
# tensor([ 4, 10, 18])

# other operations:
tensor_1 + tensor_2
# tensor([5, 7, 9])

tensor_1 - tensor_2
# tensor([-3, -3, -3])

tensor_1 / tensor_2
# tensor([0.2500, 0.4000, 0.5000])
