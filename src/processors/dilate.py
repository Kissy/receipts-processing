import cv2


def dilate(_original_image, image, kernel_size):
    morph_structure = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.dilate(image, morph_structure)
