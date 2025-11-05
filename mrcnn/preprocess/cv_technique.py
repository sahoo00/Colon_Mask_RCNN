import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# img = cv2.imread('mrcnn/dataset/1.png')
# edges = cv2.Canny(img, 200, 600)
#
# plt.subplot(121),plt.imshow(img)
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges)
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
#
# plt.show()

from skimage import feature, io, morphology
import numpy as np
import matplotlib.pyplot as plt
import cv2
# Generate noisy image of a square

im = io.imread('dataset/1.png', 1)
# First trial with the Canny filter, with the default smoothing
# Increase the smoothing for better results
edges2 = feature.canny(im, sigma=7, low_threshold=0.05, high_threshold=.15)
edges2 = morphology.binary_closing(edges2)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))

ax1.imshow(im, cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title('noisy image', fontsize=20)

ax2.imshow(edges2, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('Canny filter, $\sigma=1$', fontsize=20)

fig.tight_layout()

plt.show()

img = cv2.imread('mrcnn/dataset/new_crypts/train/Annotation/1005466.svs (1, 14781, 32818, 1151, 944)_1.png', 0)
ret, thresh = cv2.threshold(img, 127, 255, 0)
im2,contours,hierarchy = cv2.findContours(thresh, 1, 2)
cnt = contours[0]
ellipse = cv2.fitEllipse(cnt)
plt.imshow(cv2.ellipse(img,ellipse,(255,255,0),20))
# plt.imshow(cv2.rotate(img, -ellipse[2]))
plt.show()




def reinhard_mat(img):
    refimg = cv2.imread("mrcnn/dataset/new_data/Images/1005465 (1, 4390, 30110, 1327, 688).png")
    refimg = refimg.astype('uint8')
    im_lab = cv2.cvtColor(refimg, cv2.COLOR_BGR2LAB)
    target_mu = np.zeros(3)
    target_sigma = np.zeros(3)
    src_mu = np.zeros(3)
    src_sigma = np.zeros(3)

    for i in range(3):
        target_mu[i] = im_lab[:, :, i].mean()
        target_sigma[i] = (im_lab[:, :, i] - target_mu[i]).std()

    im_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    for i in range(3):
        src_mu[i] = im_lab[:, :, i].mean()
        src_sigma[i] = (im_lab[:, :, i] - src_mu[i]).std()

    for i in range(3):
        im_lab[:, :, i] = (im_lab[:, :, i] - src_mu[i]) / src_sigma[i] * target_sigma[i] + target_mu[i]

    im_norm = cv2.cvtColor(im_lab, cv2.COLOR_LAB2BGR)
    return im_norm


img = cv2.imread('mrcnn/dataset/new_crypts/train/Images/1005466.svs (1, 14781, 32818, 1151, 944).png')

normalizedImg = np.zeros(img.shape)
normalizedImg = cv2.normalize(img,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
cv2.imshow('dst_rt', normalizedImg)

img = img.astype('uint8')
normalized_img = reinhard_mat(img)
plt.imshow(img)
plt.show()