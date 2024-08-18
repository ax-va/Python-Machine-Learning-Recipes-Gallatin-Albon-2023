import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Define the data preprocessing steps
transform = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('../mnist-data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('../mnist-data', train=False, transform=transform)