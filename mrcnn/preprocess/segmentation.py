import numpy as np
from skimage import feature, io, morphology
import numpy as np

import cv2
from matplotlib import pyplot as plt
img = cv2.imread('/Users/mahdi/Desktop/1005479.svs (1, 63517, 19793, 927, 1727)1.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


def show_image(im1, im2):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
    ax1.imshow(im1)
    ax2.imshow(im2)
    plt.show()
# noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
show_image(thresh, opening)
# sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)
show_image(thresh, sure_bg)
# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
ret, sure_fg = cv2.threshold(dist_transform, 0.06 * dist_transform.max(), 255, 0)
show_image(thresh, sure_fg)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
show_image(thresh, sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)
show_image(img, unknown)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1
# Now, mark the region of unknown with zero

markers[unknown == 255] = 0

markers = cv2.watershed(img, markers)
img[markers == 25] = [0, 255, 0]

show_image(img, markers)