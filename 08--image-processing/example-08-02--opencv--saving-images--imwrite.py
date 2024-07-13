import cv2

# Load image as grayscale
image = cv2.imread("../images/plane.jpg", cv2.IMREAD_GRAYSCALE)

# Save image.
# If the image already exists, imwrite() overwrites it.
cv2.imwrite("../images/plane_grayscale.jpg", image)
# True
