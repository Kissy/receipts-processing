import cv2


def otsu_threshold(_original_image, image):
    (_, new_image) = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return new_image
