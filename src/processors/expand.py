import cv2


def expand(_original_image, image, border_size):
    return cv2.copyMakeBorder(image, top=border_size, bottom=border_size, left=border_size, right=border_size,
                                borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
