import cv2
import numpy as np


def load_image(filename: str) -> np.ndarray:
    """
    Loads image in RGB from filename.

    Args:
        filename (str): filename to load
    Returns:
        np.ndarray: loaded image in RGB
    """
    image_bgr = cv2.imread(filename, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb


if __name__ == "__main__":
    image = load_image("../../images/plane.jpg")
