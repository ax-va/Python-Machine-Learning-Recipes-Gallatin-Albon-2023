"""
Train an image classification neural network.
->
Use a convolutional neural network in PyTorch.

Convolutional neural networks typically consist of:

- *convolutional layers* to learn important image features;

- *pooling layer* typically to reduce the dimensionality of the inputs from the previous layer
by performing *max pooling* (selecting the pixel in the filter with the highest value)
or *average pooling* (taking an average of the input pixels to use instead);

- *fully connected* layer to create a binary classification task using some activation function.
"""
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Define the convolutional neural network architecture
class ConvolutionalNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            1, 32,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            32, 64,
            kernel_size=3,
            padding=1,
        )
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(self.dropout1(x), kernel_size=2)
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(self.dropout2(x)))
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)


# Set the device to run on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the data preprocessing steps
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ]
)

# Load the MNIST dataset and use as training data
train_dataset = datasets.MNIST('../mnist-data', train=True, transform=transform, download=True)
# Use loaded data as test data
test_dataset = datasets.MNIST('../mnist-data', train=False, transform=transform)

# Create data loaders
batch_size = 64
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True,
)

# Initialize the model and optimizer
model = ConvolutionalNN().to(device)
optimizer = Adam(model.parameters())
# Compile the model using torch 2.0's optimizer
model = torch.compile(model)

# Define the training loop
model.train()
for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = nn.functional.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    print("batch_idx:", batch_idx)
    # batch_idx: 1
    # ...
    # batch_idx: 937

# Define the testing loop
model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        # Get the index of the max log-probability
        test_loss += nn.functional.nll_loss(
            output, target,
            reduction='sum',  # Sum up batch loss
        ).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)
print("test_loss:", test_loss)
# test_loss: 0.046513942152261734
