import cv2
from matplotlib import pyplot as plt

# Load image as grayscale.
# Intensity values range from black = 0 to white = 255.
image = cv2.imread("../images/plane.jpg", cv2.IMREAD_GRAYSCALE)

# Show image
plt.imshow(image, cmap="gray")
plt.axis("off")
plt.show()

# Show data type
type(image)
# numpy.ndarray

# Show image data
image
# array([[140, 136, 146, ..., 132, 139, 134],
#        [144, 136, 149, ..., 142, 124, 126],
#        [152, 139, 144, ..., 121, 127, 134],
#        ...,
#        [156, 146, 144, ..., 157, 154, 151],
#        [146, 150, 147, ..., 156, 158, 157],
#        [143, 138, 147, ..., 156, 157, 157]], dtype=uint8)

# Get resolution of 3600 x 2270
image.shape
# (2270, 3600)

# Load image in BGR (blue, green, red) colors
image_bgr = cv2.imread("../images/plane.jpg", cv2.IMREAD_COLOR)

# Show pixel
image_bgr[0, 0]
# array([195, 144, 111], dtype=uint8)

# OpenCV uses BGR as default.
# Matplotlib uses RGB as default.
# In order to show the image in Matplotlib, we should convert it to RGB.
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Show image
plt.imshow(image_rgb), plt.axis("off")
plt.show()
