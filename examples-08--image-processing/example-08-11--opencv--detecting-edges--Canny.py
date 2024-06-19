"""
Find the edges in an image.
->
Use the Canny edge detector.

The Canny detector requires two parameters denoting low and high gradient threshold values:
- pixels between the low and high thresholds -> weak edge pixels.
- pixels above the high threshold -> strong edge pixels.

Other edge detection techniques: Sobel filters, Laplacian edge detector, etc.
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image as grayscale
image_gray = cv2.imread("../images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# Calculate median intensity
median_intensity = np.median(image_gray)

# Set thresholds to be one standard deviation above and below median intensity
lower_threshold = int(max(0, (1.0 - 0.33) * median_intensity))
upper_threshold = int(min(255, (1.0 + 0.33) * median_intensity))
# We can get better results determining a good pair of low
# and high threshold values through manual trial and error.

# Apply Canny edge detector
image_canny = cv2.Canny(image_gray, lower_threshold, upper_threshold)

# Show image
plt.imshow(image_canny, cmap="gray"), plt.axis("off")
plt.show()

cv2.imwrite(
    'example-08-11--opencv--detecting-edges--Canny-1.jpg',
    image_canny,
)
