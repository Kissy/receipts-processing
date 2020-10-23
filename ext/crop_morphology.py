#!/usr/bin/env python
"""Crop an image to just the portions containing text.

Usage:

    ./crop_morphology.py path/to/image.jpg

This will place the cropped image in path/to/image.crop.png.

For details on the methodology, see
http://www.danvk.org/2015/01/07/finding-blocks-of-text-in-an-image-using-python-opencv-and-numpy.html
"""
import functools
import glob
import os
import sys
import random
import cv2
import numpy as np
from scipy.ndimage.filters import rank_filter
import processors


def dilate(image, n, iterations):
    """Dilate using an NxN '+' sign shape. ary is np.uint8."""
    kernel = np.zeros((n, n), dtype=np.uint8)
    kernel[(n - 1) // 2, :] = 1
    dilated_image = cv2.dilate(image // 255, kernel, iterations=iterations)

    kernel = np.zeros((n, n), dtype=np.uint8)
    kernel[:, (n - 1) // 2] = 1
    dilated_image = cv2.dilate(dilated_image, kernel, iterations=iterations)
    return dilated_image


def props_for_contours(contours, ary):
    """Calculate bounding box & the number of set pixels for each contour."""
    c_info = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        c_im = np.zeros(ary.shape)
        cv2.drawContours(c_im, [c], 0, 255, -1)
        c_info.append({
            'x1': x,
            'y1': y,
            'x2': x + w - 1,
            'y2': y + h - 1,
            'sum': np.sum(ary * (c_im > 0)) / 255
        })
    return c_info


def union_crops(crop1, crop2):
    """Union two (x1, y1, x2, y2) rects."""
    x11, y11, x21, y21 = crop1
    x12, y12, x22, y22 = crop2
    return min(x11, x12), min(y11, y12), max(x21, x22), max(y21, y22)


def intersect_crops(crop1, crop2):
    x11, y11, x21, y21 = crop1
    x12, y12, x22, y22 = crop2
    return max(x11, x12), max(y11, y12), min(x21, x22), min(y21, y22)


def crop_area(crop):
    x1, y1, x2, y2 = crop
    return max(0, x2 - x1) * max(0, y2 - y1)


def find_border_components(contours, edges):
    borders = []
    area = edges.shape[0] * edges.shape[1]
    for i, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        if w * h > 0.5 * area:
            borders.append((i, x, y, x + w - 1, y + h - 1))

    return borders


def angle_from_right(deg):
    return min(deg % 90, 90 - (deg % 90))


def remove_border(contour, edges):
    """Remove everything outside a border contour."""
    # Use a rotated rectangle (should be a good approximation of a border).
    # If it's far from a right angle, it's probably two sides of a border and
    # we should use the bounding box instead.
    c_im = np.zeros(edges.shape)
    r = cv2.minAreaRect(contour)
    degs = r[2]
    if angle_from_right(degs) <= 10.0:
        box = cv2.boxPoints(r)
        box = np.int0(box)
        cv2.drawContours(c_im, [box], 0, 255, -1)
        cv2.drawContours(c_im, [box], 0, 0, 4)
    else:
        x1, y1, x2, y2 = cv2.boundingRect(contour)
        cv2.rectangle(c_im, (x1, y1), (x2, y2), 255, -1)
        cv2.rectangle(c_im, (x1, y1), (x2, y2), 0, 4)

    return np.minimum(c_im, edges)


def find_components(edges):
    """Dilate the image until there are just a few connected components.

    Returns contours for these components."""
    # Perform increasingly aggressive dilation until there are just a few
    # connected components.
    contours = None
    count = 21
    n = 1
    while count > 16:
        n += 1
        dilated_image = dilate(edges, n=3, iterations=n)
        hierarchy, contours, offset = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        count = len(contours)

    return contours


def find_optimal_components_subset(contours, edges):
    """Find a crop which strikes a good balance of coverage/compactness.

    Returns an (x1, y1, x2, y2) tuple.
    """
    c_info = props_for_contours(contours, edges)
    c_info.sort(key=lambda x: -x['sum'])
    total = np.sum(edges) / 255
    area = edges.shape[0] * edges.shape[1]

    c = c_info[0]
    del c_info[0]
    this_crop = c['x1'], c['y1'], c['x2'], c['y2']
    crop = this_crop
    covered_sum = c['sum']

    while covered_sum < total:
        changed = False
        recall = 1.0 * covered_sum / total
        prec = 1 - 1.0 * crop_area(crop) / area
        f1 = 2 * (prec * recall / (prec + recall))
        # print '----'
        for i, c in enumerate(c_info):
            this_crop = c['x1'], c['y1'], c['x2'], c['y2']
            new_crop = union_crops(crop, this_crop)
            new_sum = covered_sum + c['sum']
            new_recall = 1.0 * new_sum / total
            new_prec = 1 - 1.0 * crop_area(new_crop) / area
            new_f1 = 2 * new_prec * new_recall / (new_prec + new_recall)

            # Add this crop if it improves f1 score,
            # _or_ it adds 25% of the remaining pixels for <15% crop expansion.
            # ^^^ very ad-hoc! make this smoother
            remaining_frac = c['sum'] / (total - covered_sum)
            new_area_frac = 1.0 * crop_area(new_crop) / crop_area(crop) - 1
            if new_f1 > f1 or (
                            remaining_frac > 0.25 and new_area_frac < 0.15):
                print('%d %s -> %s / %s (%s), %s -> %s / %s (%s), %s -> %s' % (
                    i, covered_sum, new_sum, total, remaining_frac,
                    crop_area(crop), crop_area(new_crop), area, new_area_frac,
                    f1, new_f1))
                crop = new_crop
                covered_sum = new_sum
                del c_info[i]
                changed = True
                break

        if not changed:
            break

    return crop


def pad_crop(crop, contours, edges, border_contour, pad_px=15):
    """Slightly expand the crop to get full contours.

    This will expand to include any contours it currently intersects, but will
    not expand past a border.
    """
    bx1, by1, bx2, by2 = 0, 0, edges.shape[0], edges.shape[1]
    if border_contour is not None and len(border_contour) > 0:
        c = props_for_contours([border_contour], edges)[0]
        bx1, by1, bx2, by2 = c['x1'] + 5, c['y1'] + 5, c['x2'] - 5, c['y2'] - 5

    def crop_in_border(local_crop):
        x1, y1, x2, y2 = local_crop
        x1 = max(x1 - pad_px, bx1)
        y1 = max(y1 - pad_px, by1)
        x2 = min(x2 + pad_px, bx2)
        y2 = min(y2 + pad_px, by2)
        return local_crop

    crop = crop_in_border(crop)

    c_info = props_for_contours(contours, edges)
    changed = False
    for c in c_info:
        this_crop = c['x1'], c['y1'], c['x2'], c['y2']
        this_area = crop_area(this_crop)
        int_area = crop_area(intersect_crops(crop, this_crop))
        new_crop = crop_in_border(union_crops(crop, this_crop))
        if 0 < int_area < this_area and crop != new_crop:
            print('%s -> %s' % (str(crop), str(new_crop)))
            changed = True
            crop = new_crop

    if changed:
        return pad_crop(crop, contours, edges, border_contour, pad_px)
    else:
        return crop


def downscale_image(im, max_dim=2048):
    """Shrink im until its longest dimension is <= max_dim.

    Returns new_image, scale (where scale <= 1).
    """
    a, b = im.shape[:2]
    if max(a, b) <= max_dim:
        return 1.0, im

    scale = 1.0 * max_dim / max(a, b)
    new_im = im.resize((int(a * scale), int(b * scale)), Image.ANTIALIAS)
    return scale, new_im


def star(f):
    return lambda args: f(*args)


def create_pipeline(filepath, *functions, debug=False):
    cache_folder = filepath.replace('.jpg', '') + "/"
    if debug:
        if not os.path.exists(cache_folder):
            os.makedirs(cache_folder)

    step = 0

    def compose_steps(second_step, first_step):
        def inner_compose(original_image, image):
            nonlocal step
            step += 1

            new_image = first_step(original_image, image)
            if debug and step > 1:
                cv2.imwrite(cache_folder + "step_" + str(step) + ".jpg", new_image)
            return second_step(original_image, new_image)

        return inner_compose
    return functools.reduce(compose_steps, reversed(functions))


def output_final_image(original_image, image):
    #cv2.imwrite("tmp2.jpg", original_image)
    #cv2.imwrite("tmp.jpg", image)
    print("done")


def process_image(path, out_path):
    cache_folder = path.replace('.jpg', '') + "/"
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)

    original_image = cv2.imread(path)

    pipeline = create_pipeline(path,
                               functools.partial(processors.canny, sigma=0.33),
                               processors.remove_background,
                               functools.partial(processors.remove_lines, rank=-5, kernel_size=(2, 20)),
                               functools.partial(processors.morph_close, kernel_size=(10, 2)),
                               functools.partial(processors.remove_small_areas, size=100),
                               functools.partial(processors.morph_gradient, kernel_size=(10, 10)),
                               functools.partial(processors.expand, border_size=110),
                               functools.partial(processors.morph_close, kernel_size=(100, 100)),
                               functools.partial(processors.crop, border_size=110),
                               processors.approximate_contour,
                               functools.partial(processors.dilate, kernel_size=(20, 20)),
                               functools.partial(processors.expand, border_size=210),
                               functools.partial(processors.morph_close, kernel_size=(200, 200)),
                               functools.partial(processors.crop, border_size=210),
                               processors.approximate_contour,
                               processors.apply_mask,
                               #functools.partial(processors.morph_gradient, kernel_size=(5, 5)),
                               #functools.partial(processors.morph_close, kernel_size=(100, 100)),
                               output_final_image,
                               debug=True)
    #pipeline(original_image, original_image)

    test = True
    if test == True:
        return

    scale, downscaled_image = downscale_image(original_image)

    edges = cv2.Canny(downscaled_image, 100, 200)
    #cv2.imwrite(cache_folder + "step_1.jpg", edges)

    #edges = auto_canny(downscaled_image)
    #cv2.imwrite(cache_folder + "step_1.jpg", edges)

    # # TODO: dilate image _before_ finding a border. This is crazy sensitive!
    # hierarchy, contours, offset = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # borders = find_border_components(contours, edges)
    # borders.sort(key=star(lambda i, x1, y1, x2, y2: (x2 - x1) * (y2 - y1)))
    #
    # border_contour = None
    # if len(borders):
    #     border_contour = contours[borders[0][0]]
    #     edges = remove_border(border_contour, edges)
    #
    # edges = 255 * (edges > 0).astype(np.uint8)
    # cv2.imwrite(cache_folder + "step_2.jpg", edges)
    #
    # # Remove ~1px borders using a rank filter.
    # maxed_rows = rank_filter(edges, -4, size=(1, 20))
    # maxed_cols = rank_filter(edges, -4, size=(20, 1))
    # without_borders = np.minimum(np.minimum(edges, maxed_rows), maxed_cols)
    # edges = without_borders
    # cv2.imwrite(cache_folder + "step_3.jpg", edges)

    contours = find_components(edges)
    if len(contours) == 0:
        print('%s -> (no text!)' % path)
        return

    for c in contours:
        cv2.drawContours(original_image, [c], 0, 255, 2)
    cv2.imwrite(cache_folder + "step_4.jpg", original_image)

    crop = find_optimal_components_subset(contours, edges)
    crop = pad_crop(crop, contours, edges, border_contour)

    crop = [int(x / scale) for x in crop]  # upscale to the original image size.
    #text_im = original_image[crop[1]:crop[3], crop[0]:crop[2]]
    #cv2.imwrite(out_path, original_image)
    print('%s -> %s' % (path, out_path))


if __name__ == '__main__':
    if len(sys.argv) == 2 and '*' in sys.argv[1]:
        files = glob.glob(sys.argv[1])
        random.shuffle(files)
    else:
        files = sys.argv[1:]

    for path in files:
        out_path = path.replace('.jpg', '.crop.jpg')
        #if os.path.exists(out_path):
        #    continue
        process_image(path, out_path)
