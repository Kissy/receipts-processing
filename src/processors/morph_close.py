import cv2


def morph_close(_original_image, image, kernel_size):
    morph_structure = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, morph_structure)
