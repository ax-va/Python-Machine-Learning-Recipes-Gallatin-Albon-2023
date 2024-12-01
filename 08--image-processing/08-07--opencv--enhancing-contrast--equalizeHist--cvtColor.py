"""
Increase the contrast between pixels in an image.
->
Histogram equalization is a tool for standing out image objects and shapes.
That is able to make objects of interest more distinguishable from other objects or backgrounds.

For images in grayscale:
- equalizeHist

For color images:
- Convert to YUV (Y = luma, or brightness, U, V = colors).
->
- equalizeHist
->
- Convert to BGR or RGB.
"""
import cv2
from matplotlib import pyplot as plt

# grayscale
image = cv2.imread("../images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# Enhance image
image_enhanced = cv2.equalizeHist(image)

# Show image
plt.imshow(image_enhanced, cmap="gray"), plt.axis("off")
plt.show()

cv2.imwrite(
    'example-08-07-1--opencv--enhancing-contrast--equalizeHist--cvtColor.jpg',
    image_enhanced,
)

# color
image_bgr = cv2.imread("../images/plane.jpg")

# Convert to YUV
image_yuv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YUV)

# Apply histogram equalization
image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])

# Convert to RGB
image_rgb = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)

# Show image
plt.imshow(image_rgb), plt.axis("off")
plt.show()

# Transform RGB to BGR and save the color image
cv2.imwrite(
    'example-08-07-2--opencv--enhancing-contrast--equalizeHist--cvtColor.jpg',
    cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
)
