import cv2
import numpy
import transform


def approximate_contour(_original_image, image):
    hierarchy, contours, offset = cv2.findContours(image.astype(numpy.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask = numpy.zeros(image.shape).astype(numpy.uint8)

    for contour in contours:
        points = cv2.convexHull(contour)
        cv2.drawContours(mask, [points], 0, 255, -1)

    # TODO make another step
    #morph_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    #mask = cv2.dilate(mask, morph_structure)

    # TODO make another step
    #median = numpy.median(_original_image)
    #original_image[mask == 0] = 255



    # TODO make another step
    #gray = cv2.cvtColor(_original_image, cv2.COLOR_RGB2GRAY)
    #(_, image) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 251, 20)
    return image.astype(numpy.uint8)
