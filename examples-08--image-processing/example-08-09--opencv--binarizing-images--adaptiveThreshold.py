"""
Binarize = convert a greyscale image to its black and white form.

->

Adaptive thresholding = the threshold value for a pixel
is determined by the pixel intensities of its neighbors.

->

denoising an image = keeping only the most important elements

For example, thresholding is applied to photos
of printed text to isolate the letters from the page.
"""
import cv2
from matplotlib import pyplot as plt

# Load image as grayscale
image_grey = cv2.imread("../images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# Apply adaptive thresholding
max_output_value = 255  # the maximum intensity of the output pixel intensities
neighborhood_size = 99  # the size of the neighborhood used to determine a pixel's threshold
subtract_from_mean = 10  # a constant subtracted from the calculated threshold -> fine-tuning

# Apply cv2.ADAPTIVE_THRESH_GAUSSIAN_C:
# pixel's threshold to be a weighted sum of the neighboring pixel intensities
# ->
# weights determined by a Gaussian window
image_gaussian_threshold = cv2.adaptiveThreshold(
    image_grey,
    max_output_value,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    neighborhood_size,
    subtract_from_mean,
)

# # Show image
# plt.imshow(image_gaussian_threshold, cmap="gray"), plt.axis("off")
# plt.show()

cv2.imwrite('example-08-09--opencv--binarizing-images--adaptiveThreshold-1.jpg', image_gaussian_threshold)

# Apply cv2.ADAPTIVE_THRESH_MEAN_C:
# the mean of the neighboring pixels
image_mean_threshold = cv2.adaptiveThreshold(
    image_grey,
    max_output_value,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    neighborhood_size,
    subtract_from_mean,
)

# # Show image
# plt.imshow(image_mean_threshold, cmap="gray"), plt.axis("off")
# plt.show()

cv2.imwrite('example-08-09--opencv--binarizing-images--adaptiveThreshold-2.jpg', image_mean_threshold)
