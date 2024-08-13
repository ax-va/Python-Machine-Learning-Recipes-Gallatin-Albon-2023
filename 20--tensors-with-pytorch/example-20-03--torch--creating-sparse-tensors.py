"""
Efficiently represent data with very few nonzero values with a tensor.
-> Use PyTorch's to_sparse.

See also:
- PyTorch: Sparse Tensor
https://pytorch.org/docs/stable/sparse.html
"""
import torch

# Create a tensor
tensor = torch.tensor(
    [
        [0, 0],
        [0, 1],
        [3, 0],
    ]
)
# tensor([[0, 0],
#         [0, 1],
#         [3, 0]])

# Create a sparse tensor from a regular tensor
sparse_tensor = tensor.to_sparse()
# tensor(indices=tensor([[1, 2],
#                        [1, 0]]),
#        values=tensor([1, 3]),
#        size=(3, 2), nnz=2, layout=torch.sparse_coo)

# "tensor" and "sparse_tensor" are of the same type
# that is different from NumPy and SciPy's sparse.csr_matrix.
type(tensor)
# torch.Tensor
type(sparse_tensor)
# torch.Tensor
