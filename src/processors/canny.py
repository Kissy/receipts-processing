import numpy
import cv2


def canny(_original_image, image, sigma):
    # compute the median of the single channel pixel intensities
    v = numpy.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    #old = cv2.Canny(image, 100, 200)
    #numpy.hstack([old, edged]) # TODO TEST

    return edged
