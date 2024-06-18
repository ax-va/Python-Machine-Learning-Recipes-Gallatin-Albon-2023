"""
Isolate the foreground of an image.
->
Use the GrabCut algorithm applied to a rectangle around the desired foreground.

GrabCut assumes everything outside this rectangle to be background and
uses that information to figure out what is likely background inside the rectangle.

See also:
https://grabcut.weebly.com/background--algorithm.html
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image and convert to RGB
image_bgr = cv2.imread('../images/plane_256x256.jpg')
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
# array([[[107, 163, 212],
#         [105, 161, 210],
#         [106, 159, 209],
#         ...,
#         [ 93, 154, 208],
#         [ 93, 154, 208],
#         [ 93, 154, 208]],
# ...

image_rgb.shape
# (256, 256, 3)

image_rgb.shape[:2]
# (256, 256)

# Rectangle values: start x, start y, width, height
rectangle = (0, 56, 256, 150)

# Create initial mask
mask = np.zeros(image_rgb.shape[:2], np.uint8)
# array([[0, 0, 0, ..., 0, 0, 0],
#        [0, 0, 0, ..., 0, 0, 0],
#        [0, 0, 0, ..., 0, 0, 0],
#        ...,
#        [0, 0, 0, ..., 0, 0, 0],
#        [0, 0, 0, ..., 0, 0, 0],
#        [0, 0, 0, ..., 0, 0, 0]], dtype=uint8)

# Create temporary arrays used by grabCut
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# Apply grabCut
cv2.grabCut(
    image_rgb,  # image
    mask,
    rectangle,
    bgdModel,  # temporary array for background
    fgdModel,  # temporary array for foreground
    5,  # number of iterations
    cv2.GC_INIT_WITH_RECT,  # Initiative with rectangle
)
#  array([[ 3.58154888e-01,  8.87617088e-02,  3.07697254e-01,
#           1.13518959e-01,  1.31867190e-01,  7.96318152e+01,
#           1.42454486e+02,  1.96163325e+02,  1.40773485e+02,
#           1.71938179e+02,  2.02640425e+02,  8.75006618e+01,
#           1.47160084e+02,  1.98042076e+02,  1.57438822e+02,
#           1.80246601e+02,  2.04860838e+02,  1.07784298e+02,
#           1.56056079e+02,  1.99441645e+02,  3.09994177e+00,
#           1.84301846e+00,  1.62402157e+00,  1.84301846e+00,
#           1.54184190e+00,  1.25685013e+00,  1.62402157e+00,
#           1.25685013e+00,  1.19260792e+00,  5.27600890e+01,
#           2.06015484e+01, -2.68975642e+00,  2.06015484e+01,
#           9.12440808e+00, -5.92622844e-01, -2.68975642e+00,
#          -5.92622844e-01,  3.30876427e+00,  3.16907519e+01,
#           1.60270623e+01,  1.08697040e+01,  1.60270623e+01,
#           1.18544150e+01,  8.73398185e+00,  1.08697040e+01,
#           8.73398185e+00,  9.72473606e+00,  3.29471636e+01,
#           1.37922770e+01,  4.86997954e+00,  1.37922770e+01,
#           6.46184645e+00,  2.61003495e+00,  4.86997954e+00,
#           2.61003495e+00,  3.11413102e+00,  1.03842777e+02,
#           6.22674605e+01,  4.21094055e+01,  6.22674605e+01,
#           5.60925963e+01,  5.97935033e+01,  4.21094055e+01,
#           5.97935033e+01,  8.70934738e+01]]),
#  array([[ 1.10787481e-01,  4.91235503e-01,  6.42906318e-02,
#           8.17137107e-02,  2.51972674e-01,  1.87519120e+02,
#           1.99339866e+02,  2.08411090e+02,  1.87338292e+01,
#           3.38337646e+01,  3.88560802e+01,  9.80716639e+01,
#           1.48253707e+02,  1.94682867e+02,  1.47928710e+02,
#           1.66134802e+02,  1.81419313e+02,  7.41319882e+01,
#           9.19571248e+01,  1.03572720e+02,  2.26931279e+02,
#           2.00957890e+02,  1.51364989e+02,  2.00957890e+02,
#           2.10070437e+02,  1.94049099e+02,  1.51364989e+02,
#           1.94049099e+02,  2.35460068e+02,  1.27453194e+02,
#           1.39865520e+02,  1.17164259e+02,  1.39865520e+02,
#           1.63677411e+02,  1.42650288e+02,  1.17164259e+02,
#           1.42650288e+02,  1.42496212e+02,  1.93854008e+02,
#           9.48681446e+01,  4.23233860e+01,  9.48681446e+01,
#           6.03359624e+01,  3.82435561e+01,  4.23233860e+01,
#           3.82435561e+01,  4.64422600e+01,  1.61039636e+02,
#           5.85540041e+01, -7.41321293e+01,  5.85540041e+01,
#           6.21360733e+01,  6.29752319e+01, -7.41321293e+01,
#           6.29752319e+01,  2.44240897e+02,  1.41964084e+03,
#           1.31252731e+03,  1.20907846e+03,  1.31252731e+03,
#           1.28507088e+03,  1.22846865e+03,  1.20907846e+03,
#           1.22846865e+03,  1.22734559e+03]]))

np.unique(mask)
# array([0, 2, 3], dtype=uint8)

# Create new mask where sure (mask == 0) and likely (mask == 2) backgrounds set to 0, otherwise 1
new_mask = np.where((mask == 0) | (mask == 2), 0, 1).astype('uint8')
# array([[0, 0, 0, ..., 0, 0, 0],
#        [0, 0, 0, ..., 0, 0, 0],
#        [0, 0, 0, ..., 0, 0, 0],
#        ...,
#        [0, 0, 0, ..., 0, 0, 0],
#        [0, 0, 0, ..., 0, 0, 0],
#        [0, 0, 0, ..., 0, 0, 0]], dtype=uint8)
new_mask.shape
# (256, 256)
np.unique(new_mask)
# array([0, 1], dtype=uint8)

# Show mask
plt.imshow(mask, cmap='gray'), plt.axis("off")
plt.show()
# The black region is the area outside the rectangle that is assumed to be background.
# The gray area is what GrabCut considered likely background,
# while the white area is likely foreground.

mask_mapped = np.zeros(mask.shape, np.uint8)
mask_mapped[mask == 0] = 0  # black
mask_mapped[mask == 2] = 127  # gray
mask_mapped[mask == 3] = 255  # white
cv2.imwrite(
    'example-08-10--opencv--removing-backgrounds--grabCut-1.jpg',
    mask_mapped,
)

# Show new_mask: merged black and gray regions
plt.imshow(new_mask, cmap='gray'), plt.axis("off")
plt.show()

cv2.imwrite(
    'example-08-10--opencv--removing-backgrounds--grabCut-2.jpg',
    new_mask * 255,
)
# 0 * 255 = 0 (black)
# 1 * 255 = 255 (white)

# Apply new_mask to the image so that only the foreground remains.
# ->
# Multiply image with new mask to subtract background.
new_mask[:, :, np.newaxis]
# array([[[0],
#         [0],
#         [0],
#         ...,
#         [0],
#         [0],
#         [0]],
# ...
image_rgb_nobg = image_rgb * new_mask[:, :, np.newaxis]

# Show image
plt.imshow(image_rgb_nobg), plt.axis("off")
plt.show()

cv2.imwrite(
    'example-08-10--opencv--removing-backgrounds--grabCut-3.jpg',
    cv2.cvtColor(image_rgb_nobg, cv2.COLOR_RGB2BGR),
)
