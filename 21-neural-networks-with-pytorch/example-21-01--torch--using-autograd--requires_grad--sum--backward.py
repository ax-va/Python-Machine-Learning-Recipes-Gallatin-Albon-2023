"""
Use PyTorch's autograd features to compute and store
the gradients after undergoing forward and back propagations.
->
Create tensors with "requires_grad=True".

PyTorch provides an ability to automatically compute gradients.
PyTorch uses a directed acyclic graph (DAG) to keep a record of all data
and computational operations being performed on that data.
Calling the detach() method on the tensor with gradient computation converts
it into a NumPy array with breaking the graph and the ability
to automatically compute gradients.

See also:
- PyTorch: A Gentle Introduction to torch.autograd
https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
"""
import torch

torch.__version__
# '2.3.1+cu121'

# Create a torch tensor that requires gradients
tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
# tensor([1., 2., 3.], requires_grad=True)

tensor.grad
# None

# Perform a tensor operation simulating "forward propagation"
tensor_sum = tensor.sum()
# tensor(6., grad_fn=<SumBackward0>)

# Perform back propagation
tensor_sum.backward()

# View the gradients
tensor.grad
# tensor([1., 1., 1.])
