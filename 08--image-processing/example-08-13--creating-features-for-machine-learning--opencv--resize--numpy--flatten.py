"""
Convert an image into an observation for machine learning.

Exploding the features for large images.
->
The number of features might far exceed the number of observations.
->
dimensionality reduction needed
"""
import cv2

image = cv2.imread("../images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# Resize image to 10 pixels by 10 pixels
image_10x10 = cv2.resize(image, (10, 10))
image_10x10.shape
# (10, 10)

# Convert image data to one-dimensional vector
image_10x10.flatten()
# array([133, 130, 130, 129, 130, 129, 129, 128, 128, 127, 135, 131, 131,
#        131, 130, 130, 129, 128, 128, 128, 134, 132, 131, 131, 130, 129,
#        129, 128, 130, 133, 132, 158, 130, 133, 130,  46,  97,  26, 132,
#        143, 141,  36,  54,  91,   9,   9,  49, 144, 179,  41, 142,  95,
#         32,  36,  29,  43, 113, 141, 179, 187, 141, 124,  26,  25, 132,
#        135, 151, 175, 174, 184, 143, 151,  38, 133, 134, 139, 174, 177,
#        169, 174, 155, 141, 135, 137, 137, 152, 169, 168, 168, 179, 152,
#        139, 136, 135, 137, 143, 159, 166, 171, 175], dtype=uint8)
image_10x10.flatten().shape
# (100,)

# Load image in color
image_color = cv2.imread("../images/plane_256x256.jpg", cv2.IMREAD_COLOR)

# Resize image to 10 pixels by 10 pixels
image_color_10x10 = cv2.resize(image_color, (10, 10))

# Convert image data to one-dimensional vector, show dimensions
image_color_10x10.flatten().shape
# (300,)

# Exploding the features for large images.
# ->
# The number of features might far exceed the number of observations.
# ->
# dimensionality reduction needed

# Load image in grayscale
image_256x256_gray = cv2.imread("../images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# Convert image data to one-dimensional vector, show dimensions
image_256x256_gray.flatten().shape
# (65536,)

# Load image in color
image_256x256_color = cv2.imread("../images/plane_256x256.jpg", cv2.IMREAD_COLOR)

# Convert image data to one-dimensional vector, show dimensions
image_256x256_color.flatten().shape
# (196608,)
