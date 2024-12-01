"""
Apply an operation to all elements in a tensor.
->
Take advantage of *broadcasting* with PyTorch.

Notices:
- Basic operations will take advantage of broadcasting
to parallelize them using accelerated hardware such as GPUs.

- PyTorch doesn't include a vectorize method for
applying a function over all elements in a tensor.

See also:
- PyTorch: Broadcasting semantics
https://pytorch.org/docs/stable/notes/broadcasting.html

- PyTorch: Vectorization and Broadcasting with PyTorch
https://blog.paperspace.com/pytorch-vectorization-and-broadcasting/
"""
import torch

# Create a tensor
tensor = torch.tensor([1, 2, 3])
# tensor([1, 2, 3])

# Broadcast an arithmetic operation to all elements in a tensor
tensor * 100
# tensor([100, 200, 300])

tensor + 100
# tensor([101, 102, 103])
