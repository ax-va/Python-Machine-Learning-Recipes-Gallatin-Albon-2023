"""
Detect objects in images using pretrained cascade classifiers with OpenCV.
->
Download models of Haar cascade classifiers for OpenCV:
https://github.com/opencv/opencv/tree/4.x/data/haarcascades

See also:
- Cascade Classifier
https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html
"""
import cv2
from matplotlib import pyplot as plt
from utils.image_processing import load_image  # my module

face_cascade = cv2.CascadeClassifier()
face_cascade.load(
    cv2.samples.findFile(
        "models/haarcascade_frontalface_default.xml"
    )
)

# images with faces to test the face detection
filenames = [
    "../images/faces/johnny.png",
    "../images/faces/johnny_and_orlando.png",
    "../images/faces/leo_and_kate.png",
    "../images/faces/leo_and_johnny.png",
    "../images/faces/kate.png",
    "../images/faces/leo.png",
    "../images/faces/critter.png",
    "../images/faces/freddy.png",
    "../images/faces/orlando.png",
]

for i in range(len(filenames)):
    # Load image
    image_rgb = load_image(filenames[i])

    # Detect faces and draw a rectangle
    faces = face_cascade.detectMultiScale(image_rgb)
    for (x, y, w, h) in faces:
        cv2.rectangle(
            image_rgb,
            (x, y),
            (x + h, y + w),
            (0, 255, 0),
            5,
        )

    # Show the image
    plt.subplot(1, 1, 1)
    plt.imshow(image_rgb)
    # plt.show()
    plt.savefig(f'example-08-16-{i+1}--opencv--detecting-objects--face-detection--CascadeClassifier.png')
    plt.close()
