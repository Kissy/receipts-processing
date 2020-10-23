import numpy
import cv2


def apply_mask(original_image, image):
    gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    threshold_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 251, 20)
    threshold_image[image == 0] = 255
    return threshold_image
