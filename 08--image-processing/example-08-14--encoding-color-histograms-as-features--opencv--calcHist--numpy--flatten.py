"""
Create a set of features representing the colors appearing in an image.

See also:
- pandas.DataFrame.hist
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.hist.html

- Histograms - 1 : Find, Plot, Analyze !!!
https://docs.opencv.org/3.2.0/d1/db7/tutorial_py_histogram_begins.html
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

np.random.seed(0)

image_bgr = cv2.imread("../images/plane_256x256.jpg", cv2.IMREAD_COLOR)

# Convert to RGB
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Create a list for feature values
features = []

# Calculate histogram or each channel and add to feature value list
for i, channel in enumerate(["r", "g", "b"]):  # 0 = red, 1 = green, 2 = blue
    histogram = cv2.calcHist(
        [image_rgb],  # image
        [i],  # index of channel
        None,  # no mask
        [256],  # histogram size
        [0, 256])  # range
    # The x-axis represents the 256 possible channel values,
    # and the y-axis represents the number of times a particular channel value
    # appears across all pixels in an image
    plt.plot(histogram, color=channel)
    plt.xlim([0, 256])
    features.extend(histogram)

# Show plot
# plt.show()

plt.savefig('example-08-14--encoding-color-histograms-as-features--opencv--calcHist--numpy--flatten.svg')
plt.close()

# Create a vector for an observation's feature values
observation = np.array(features).flatten()

# Show the observation's value for the first five features
observation[0:5]
# array([1027.,  217.,  182.,  146.,  146.], dtype=float32)

# The top leftmost pixel of the image has the following RGB channel values
image_rgb[0, 0]
# array([107, 163, 212], dtype=uint8)
