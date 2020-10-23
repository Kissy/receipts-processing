import cv2


def morph_gradient(_original_image, image, kernel_size):
    morph_structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, morph_structure)
