import cv2
import numpy


def find_border_components(contours, edges):
    borders = []
    area = edges.shape[0] * edges.shape[1]
    for i, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        if w * h > 0.5 * area:
            borders.append((i, x, y, x + w - 1, y + h - 1))

    return borders


def remove_outside_contour(contour, image):
    """Remove everything outside a border contour."""
    new_image = numpy.zeros(image.shape)
    rectangle = cv2.minAreaRect(contour)
    if angle_from_right(rectangle[2]) <= 10.0:
        points = cv2.convexHull(contour)
        cv2.drawContours(new_image, [points], 0, 255, -1)
        cv2.drawContours(new_image, [points], 0, 0, 10)
        return numpy.minimum(new_image, image)
    else:
        return image


def star(f):
    return lambda args: f(*args)


def angle_from_right(deg):
    return min(deg % 90, 90 - (deg % 90))


def remove_background(_original_image, image):
    # TODO: dilate image _before_ finding a border. This is crazy sensitive!
    hierarchy, contours, offset = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    borders = find_border_components(contours, image)
    borders.sort(key=star(lambda i, x1, y1, x2, y2: (x2 - x1) * (y2 - y1)))

    if len(borders):
        border_contour = contours[borders[0][0]]
        edges = remove_outside_contour(border_contour, image)
        return 255 * (edges > 0).astype(numpy.uint8)
        #peri = cv2.arcLength(border_contour, True)
        #approx = cv2.approxPolyDP(border_contour, 0.02 * peri, True)
        #return transform.four_point_transform(_original_image, approx.reshape(4, 2))
        #return 255 * (edges > 0).astype(numpy.uint8)

    return image
