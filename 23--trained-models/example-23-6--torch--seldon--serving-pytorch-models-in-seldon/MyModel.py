"""
Serve a trained PyTorch model for real-time predictions.
->
Serve the model using the Seldon Core Python wrapper.

Seldon Core (with the gRPC server) is a popular framework for serving models in production.

See also:
- Seldon Core Python Package
https://docs.seldon.io/projects/seldon-core/en/latest/python/python_module.html

- PyTorch: TorchServe
https://pytorch.org/serve/
"""
import torch
import torch.nn as nn
import logging


class SequentialNN(nn.Module):
    """
    Feedforward two-layer neural network for binary classification using nn.Sequential.
    Each layer is "dense" (also called "fully connected")
    = All the units in the previous layer and in the next layer are connected.
    """
    def __init__(self):
        """ Initiates a network architecture. """
        super().__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(10, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.sequential(x)
        return x


# Create a Seldon model with the name `MyModel`
class MyModel:
    # Loads the model
    def __init__(self):
        self.network = SequentialNN()
        self.network.load_state_dict(
            torch.load("model_torch_2_3_1+cu121.pt")["model_state_dict"],
            strict=False,
        )
        logging.info(self.network.eval())

    # Makes a prediction
    def predict(self, X, features_names=None):
        return (self.network.forward(torch.from_numpy(X).float())).detach().numpy()


# Run the model in Docker
"""
$ sudo docker run -it -v $(pwd):/app -p 9000:9000 \
kylegallatin/seldon-example \
seldon-core-microservice MyModel --service-type MODEL
"""

# Post request
"""
$ curl -X POST http://127.0.0.1:9000/predict \
-H 'Content-Type: application/json' \
-d '{"data": {"ndarray":[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]}}'
"""
# output:
"""
{"data":{"names":["t:0"],"ndarray":[[0.5535200238227844]]},"meta":{}}
"""
