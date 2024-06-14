"""
Sharpen an image.
->
Create a kernel that highlights the target pixel.
Then apply it to the image using filter2D.
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image as grayscale
image = cv2.imread("../images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# Create kernel
kernel = np.array(
    [[ 0, -1,  0],
     [-1,  5, -1],
     [ 0, -1,  0]]
)

# Sharpen image
image_sharp = cv2.filter2D(image, -1, kernel)

# Show image
plt.imshow(image_sharp, cmap="gray"), plt.axis("off")
plt.show()

cv2.imwrite(
    'example-08-06--opencv--sharpening-Images--filter2D-1.jpg',
    image_sharp,
)
