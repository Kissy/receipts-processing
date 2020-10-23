import cv2


def gaussian_blur(_original_image, image, kernel_size):
    return cv2.GaussianBlur(image, kernel_size, 0)
