"""
Remove the outer portion of the image to change its dimensions.
->
Crop the image easily by slicing the array.
"""
import cv2
from matplotlib import pyplot as plt

# Load image in grayscale
image = cv2.imread("../images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# Select first half of the columns and all rows
image_cropped = image[:, :128]
image_cropped.shape
# (256, 128)

# Show image
plt.imshow(image_cropped, cmap="gray"), plt.axis("off"), plt.show()
