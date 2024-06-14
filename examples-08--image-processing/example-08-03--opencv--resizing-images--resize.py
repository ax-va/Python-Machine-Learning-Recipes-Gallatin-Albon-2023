"""
See also:
The Pillow library offers many options for resizing images for this reason
https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.resize
"""
import cv2
from matplotlib import pyplot as plt

# Load image as grayscale
image = cv2.imread("../images/plane.jpg", cv2.IMREAD_GRAYSCALE)
image.shape
# (2270, 3600)

# Resize image
image_smaller = cv2.resize(image, (3600 // 10, 2270 // 10))
image_smaller.shape
# (227, 360)

# View image
plt.imshow(image_smaller, cmap="gray"), plt.axis("off")
plt.show()

cv2.imwrite(
    'example-08-03--opencv--resizing-images--resize-1.jpg',
    image_smaller,
)

image_larger = cv2.resize(image, (3600 * 2, 2270 * 2))
image_larger.shape
# (4540, 7200)

# View image
plt.imshow(image_larger, cmap="gray"), plt.axis("off")
plt.show()

cv2.imwrite(
    'example-08-03--opencv--resizing-images--resize-2.jpg',
    image_larger,
)
