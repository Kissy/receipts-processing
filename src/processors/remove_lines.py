import numpy
from scipy.ndimage.filters import rank_filter


def remove_lines(_original_image, image, rank, kernel_size):
    maxed_rows = rank_filter(image, rank, size=(kernel_size[0], kernel_size[1]))
    maxed_cols = rank_filter(image, rank, size=(kernel_size[1], kernel_size[0]))
    return numpy.minimum(numpy.minimum(image, maxed_rows), maxed_cols)
