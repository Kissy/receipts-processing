import numpy as np
import cv2
import transform


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


image = cv2.imread('test2.png')
original = image.copy()

ratio = 1

# Gray
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
cv2.imwrite("gray.jpg", image)

# Text detection
morphStructure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
image = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, morphStructure)
cv2.imwrite("gradient.jpg", image)

(_, image) = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imwrite("grandient-bin.jpg", image)
#image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 251, 20)
#cv2.imwrite("grandient-bin.jpg", image)

morphStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, morphStructure)
cv2.imwrite("closed.jpg", image)

# Blur
image = cv2.GaussianBlur(image, (9, 9), 0)
cv2.imwrite("blurred.jpg", image)

# Sharpen
#kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
#image = cv2.filter2D(image, -1, kernel)
#cv2.imwrite("sharpened.jpg", image)

# Threshold
image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 251, 20)
cv2.imwrite("threshold.jpg", image)

# Binarize
#(_, image) = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#cv2.imwrite("binary.jpg", image)

# Add borders
row, col = image.shape[:2]
bottom = image[row-2:row, 0:col]
mean = cv2.mean(bottom)[0]

bordersize = 10
border = cv2.copyMakeBorder(image, bordersize, bordersize, bordersize, bordersize, cv2.BORDER_CONSTANT, value=[mean, mean, mean])
cv2.imwrite("bordered.jpg", image)

# Canny
#image = cv2.Canny(image, 75, 200)
#cv2.imwrite("canny.jpg", image)

# Canny auto
wide = cv2.Canny(image, 10, 200)
tight = cv2.Canny(image, 225, 250)
image = auto_canny(image)
cv2.imwrite("canny.jpg", image)

# Contours
(_, cnts, _) = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

outlined = original.copy()

# loop over the contours
contour = None
screenCnt = None
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.01 * peri, True)
    cv2.drawContours(outlined, [c], -1, (0, 0, 255), 2)

    # if our approximated contour has four points, then we
    # can assume that we have found our screen
    if len(approx) == 4:
        screenCnt = approx
        contour = c
        break


# outline
#cv2.drawContours(outlined, [contour], -1, (255, 0, 0), 1)
#cv2.drawContours(outlined, [screenCnt], -1, (0, 255, 0), 1)
cv2.imwrite("outlined.jpg", outlined)

if screenCnt:
    image = transform.four_point_transform(original, screenCnt.reshape(4, 2) * ratio)
    cv2.imwrite("unwarped.jpg", image)
else:
    image = original

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#image = threshold_adaptive(warped, 251, offset=10) # TODO test
#image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 8)
#image = warped.astype("uint8") * 255

#(_, image) = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 251, 20)
cv2.imwrite("binary2.jpg", image)


cv2.imwrite("final.jpg", image)
