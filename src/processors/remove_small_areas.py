import cv2
import numpy


def remove_small_areas(_original_image, image, size):
    (_, contours, _) = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    kept_contours = []
    new_image = numpy.zeros(image.shape)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > size:
            kept_contours.append(contour)
            cv2.drawContours(new_image, [contour], 0, 255, -1)

    return new_image