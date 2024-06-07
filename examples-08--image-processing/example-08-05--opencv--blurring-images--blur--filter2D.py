"""
Blur an image.
->
Each pixel is transformed to be the average value of its neighbors.
->
Larger kernels produce smoother images.
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image as grayscale
image = cv2.imread("../images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# Blur image with the kernel size of (5, 5)
image_blurry = cv2.blur(image, (5, 5))

# # Show image
# plt.imshow(image_blurry, cmap="gray"), plt.axis("off")
# plt.show()

cv2.imwrite('example-08-05--opencv--blurring-images--blur--filter2D-1.jpg', image_blurry)

# Blur image with the kernel size of (10, 10)
image_very_blurry = cv2.blur(image, (100, 100))

# # Show image
# plt.imshow(image_very_blurry, cmap="gray"), plt.xticks([]), plt.yticks([])
# plt.show()

cv2.imwrite('example-08-05--opencv--blurring-images--blur--filter2D-2.jpg', image_very_blurry)

# Create kernel
kernel = np.ones((5, 5)) / 25.0
# array([[0.04, 0.04, 0.04, 0.04, 0.04],
#        [0.04, 0.04, 0.04, 0.04, 0.04],
#        [0.04, 0.04, 0.04, 0.04, 0.04],
#        [0.04, 0.04, 0.04, 0.04, 0.04],
#        [0.04, 0.04, 0.04, 0.04, 0.04]])

# Apply kernel to the image manually
image_kernel = cv2.filter2D(image, -1, kernel)

# # Show image
# plt.imshow(image_kernel, cmap="gray"), plt.xticks([]), plt.yticks([])
# plt.show()

cv2.imwrite('example-08-05--opencv--blurring-images--blur--filter2D-3.jpg', image_kernel)

# Image Kernels Explained Visually
# https://setosa.io/ev/image-kernels/
