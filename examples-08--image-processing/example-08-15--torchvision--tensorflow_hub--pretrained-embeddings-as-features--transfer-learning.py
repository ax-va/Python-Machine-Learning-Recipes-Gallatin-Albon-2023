"""
Load pretrained embeddings from an existing model in PyTorch
and use them as input to one of your own models.

->

Use *transfer learning* = use pretrained image models such as ResNet.

For example, use the weights of a model trained to recognize cats
as a good start for a model to train to recognize dogs.

->

Leverage the information learned from other datasets and model
architectures without the overhead of training a model from scratch.

See also:
- Transfer Learning for Computer Vision Tutorial
https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#sphx-glr-beginner-transfer-learning-tutorial-py

- TensorFlow Hub is a repository of trained machine learning models
https://www.tensorflow.org/hub
"""
import tensorflow as tf
import tensorflow_hub as tf_hub   # Use pretrained TensorFlow models
import torch
import torchvision.models as models
from torchvision import transforms  # Use pretrained PyTorch models
from torchvision.models import ResNet18_Weights
from utils.image_processing import load_image  # my module

image_rgb = load_image("../images/plane.jpg")

# # # Using PyTorch

# Convert to pytorch data type
convert_tensor = transforms.ToTensor()
pytorch_image = convert_tensor(image_rgb)

# Load the pretrained model
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

# Select the specific layer of the model we want output from
layer = model._modules.get('avgpool')

# Set model to evaluation mode
model.eval()

# Infer the embedding with the no_grad option
with torch.no_grad():
    embedding_pytorch = model(pytorch_image.unsqueeze(0))

embedding_pytorch.shape
# torch.Size([1, 1000])

# # # Using TensorFlow

# Convert to tensorflow data type
tf_image = tf.image.convert_image_dtype([image_rgb], tf.float32)

# Create the model and get embeddings using the inception V1 model
embedding_model = tf_hub.KerasLayer("https://tfhub.dev/google/imagenet/inception_v1/feature_vector/5")
embedding_tf = embedding_model(tf_image)
embedding_tf.shape
# TensorShape([1, 1024])
