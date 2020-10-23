import numpy
import cv2


def sharpen(_original_image, image):
    kernel = numpy.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)
