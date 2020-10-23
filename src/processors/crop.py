

def crop(_original_image, image, border_size):
    shape = image.shape
    return image[border_size:shape[0] - border_size, border_size:shape[1] - border_size]
