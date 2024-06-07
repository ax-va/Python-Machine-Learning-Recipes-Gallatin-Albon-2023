"""
Define a range of colors and then apply a mask to the image.
->
Convert an image into HSV (hue, saturation, and value).
->
Define a range of values we want to isolate.
->
Create a mask for the image.
->
Apply the mask to the image using "bitwise_and" and convert to a desired output format.
"""
# Load libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image
image_bgr = cv2.imread('../images/plane_256x256.jpg')
# array([[[212, 163, 107],
#         [210, 161, 105],
#         [209, 159, 106],
#         ...,
#         [208, 154,  93],
#         [208, 154,  93],
#         [208, 154,  93]],
# ...

# Convert BGR to HSV
image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
# array([[[104, 126, 212],
#         [104, 128, 210],
#         [105, 126, 209],
#         ...,
#         [104, 141, 208],
#         [104, 141, 208],
#         [104, 141, 208]],
# ...

# Define range of blue values in HSV
lower_blue = np.array([50, 100, 50])
upper_blue = np.array([130, 255, 255])

# Create mask
mask = cv2.inRange(image_hsv, lower_blue, upper_blue)
# array([[255, 255, 255, ..., 255, 255, 255],
#        [255, 255, 255, ..., 255, 255, 255],
#        [255, 255, 255, ..., 255, 255, 255],
#        ...,
#        [255, 255, 255, ...,   0,   0,   0],
#        [255, 255, 255, ...,   0,   0,   0],
#        [255, 255, 255, ...,   0,   0,   0]], dtype=uint8)

# Mask image
image_bgr_masked = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
# array([[[212, 163, 107],
#         [210, 161, 105],
#         [209, 159, 106],
#         ...,
#         [208, 154,  93],
#         [208, 154,  93],
#         [208, 154,  93]],
# ...

# Convert BGR to RGB
image_rgb = cv2.cvtColor(image_bgr_masked, cv2.COLOR_BGR2RGB)
# array([[[107, 163, 212],
#         [105, 161, 210],
#         [106, 159, 209],
#         ...,
#         [ 93, 154, 208],
#         [ 93, 154, 208],
#         [ 93, 154, 208]],
# ...

# # Show image
# plt.imshow(image_rgb), plt.axis("off")
# plt.show()

cv2.imwrite(
    'example-08-08--opencv--isolating-colors--cvtColor--inRange--bitwise_and-1.jpg',
    cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
)


# # Show image and keep only the white areas
# plt.imshow(mask, cmap='gray'), plt.axis("off")
# plt.show()

cv2.imwrite(
    'example-08-08--opencv--isolating-colors--cvtColor--inRange--bitwise_and-2.jpg',
    mask,
)
