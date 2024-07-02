"""
Classify images using pretrained deep learning models in Pytorch.

ResNet18 is a pretrained model, an 18-layers NN that was trained on the ImageNet dataset.

See also:
- ResNet101 and ResNet152

- Models and pretrained weights
https://pytorch.org/vision/stable/models.html
"""
import cv2
import json
import numpy as np
import torch
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from utils.image_processing import load_image  # my module

# # Get imagenet classes
# import urllib.request
# url = "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json"
# with urllib.request.urlopen(url) as src:
#     imagenet_class_index = json.load(src)

# The data are loaded from
# https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json
with open('../data/imagenet_class_index.json', 'r') as f:
    imagenet_class_index = json.load(f)

# {'0': ['n01440764', 'tench'],
#  '1': ['n01443537', 'goldfish'],
#  '2': ['n01484850', 'great_white_shark'],
#  '3': ['n01491361', 'tiger_shark'],
# ...
#  '405': ["n02692877", "airship"],
# ...
#  '997': ['n13054560', 'bolete'],
#  '998': ['n13133613', 'ear'],
#  '999': ['n15075141', 'toilet_tissue']}

# Instantiate pretrained model
model = resnet18(weights=ResNet18_Weights.DEFAULT)

# Load image
image_rgb = load_image("../images/plane.jpg")

# Convert to pytorch data type
convert_tensor = transforms.ToTensor()
pytorch_image = convert_tensor(image_rgb)

# Set model to evaluation mode
model.eval()

# Make a prediction
prediction = model(pytorch_image.unsqueeze(0))

# Get the index of the highest predicted probability
_, index = torch.max(prediction, 1)
# tensor([463])

# Convert that to a percentage value
percentage = torch.nn.functional.softmax(prediction, dim=1)[0] * 100
# tensor([0.0470, 0.0467, 0.0451, 0.0201, 0.0450, 0.0697, 0.0533, 0.1191, 0.1065,
#         0.0412, 0.0286, 0.0332, 0.0528, 0.0336, 0.0279, 0.0433, 0.0339, 0.0602,
#         0.0513, 0.0462, 0.0200, 0.0397, 0.0224, 0.0987, 0.0338, 0.0237, 0.0353,
# ...
#         0.0593, 0.0531, 0.0120, 0.2067, 0.0200, 0.1242, 0.1010, 0.0407, 0.0416,
#         0.0758, 0.1098, 0.0456, 0.0306, 0.0244, 0.0088, 0.2911, 0.0623, 0.0311,
#         0.0569, 0.0246, 0.0307, 0.0130, 0.0328, 0.0497, 0.0974, 0.0488, 0.2703,
#         0.1906], grad_fn=<MulBackward0>)

print("Class:", imagenet_class_index[str(index.tolist()[0])][1])
# Class: airship
print(f"Max. probability: {percentage[index.tolist()[0]].item():.2f}%")
# Max. probability: 6.06%
