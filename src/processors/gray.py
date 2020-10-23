import numpy
import cv2


def gray(_original_image, image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
