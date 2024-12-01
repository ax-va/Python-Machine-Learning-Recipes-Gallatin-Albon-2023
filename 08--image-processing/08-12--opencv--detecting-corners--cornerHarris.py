"""
Detect the corners in an image.
->
1. Use the Harris corner detector:
- OpenCV's cornerHarris
https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=goodfeaturestotrack#cornerharris

The Harris corner detector is a commonly used method to detect the intersection of two edges.
That looks for windows (also called neighborhoods or patches) where small movements of the window
(imagine shaking the window) create big changes in the contents of the pixels inside the window.

2. Use the Shi-Tomasi corner detector that identifies a fixed number of strong corners:
- OpenCV's goodFeaturesToTrack
https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html#goodfeaturestotrack

"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# # # 1. Harris (cornerHarris)

# Load image
image_bgr = cv2.imread("../images/plane_256x256.jpg")
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
image_gray = np.float32(image_gray)

# Set corner detector parameters
block_size = 2  # size of the neighbor around each pixel
aperture = 29  # size of the Sobel kernel
free_parameter = 0.04  # free parameter: larger values correspond -> softer corners

# Detect corners
detector_responses = cv2.cornerHarris(
    image_gray,
    block_size,
    aperture,
    free_parameter,
)

# potential corners
detector_responses = cv2.dilate(detector_responses, None)

# Apply thresholding to keep only the most likely corners
threshold = 0.02
image_bgr[detector_responses > threshold * detector_responses.max()] = [255, 255, 255]  # white

# Convert to grayscale
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

# Show image
plt.imshow(image_gray, cmap="gray"), plt.axis("off")
plt.show()

cv2.imwrite(
    'example-08-12-1--opencv--detecting-corners--cornerHarris.jpg',
    image_gray,
)

# Show potential corners
plt.imshow(detector_responses, cmap='gray'), plt.axis("off")
plt.show()

# Use MinMaxScaler to scale values to the range 0 to 255
minmax_scaler = MinMaxScaler(feature_range=(0, 255))

cv2.imwrite(
    'example-08-12-2--opencv--detecting-corners--cornerHarris.jpg',
    minmax_scaler.fit_transform(detector_responses.flatten()[:, np.newaxis]).reshape(detector_responses.shape),
)

# # # 2. Shi-Tomasi (goodFeaturesToTrack)

image_bgr = cv2.imread('../images/plane_256x256.jpg')
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

# parameters
corners_to_detect = 10  # number of corners
minimum_quality_score = 0.05  # minimum quality of the corner (0 to 1)
minimum_distance = 25  # minimum Euclidean distance between corners

# Detect corners
corners = cv2.goodFeaturesToTrack(
    image_gray,
    corners_to_detect,
    minimum_quality_score,
    minimum_distance,
)

corners = np.int16(corners)

# Draw white circle at each corner
for corner in corners:
    x, y = corner[0]
    cv2.circle(image_gray, (x, y), 10, (255, 255, 255), -1)

# Show image
plt.imshow(image_gray, cmap='gray'), plt.axis("off")
plt.show()

cv2.imwrite(
    'example-08-12-3--opencv--detecting-corners--cornerHarris.jpg',
    image_gray,
)
