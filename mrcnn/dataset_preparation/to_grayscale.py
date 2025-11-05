import cv2
import glob, os, errno
import numpy as np

for fil in glob.glob("/Users/mahdi/software/Colon_Mask_RCNN/mrcnn/dataset/Normalized_Images/test/Annotation_gland/*"):
    print(fil)
    image = cv2.imread(fil, cv2.IMREAD_UNCHANGED)
    if len(image.shape) == 2:
        continue
    image = np.array([[max(x) for x in y] for y in image])
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to greyscale
    cv2.imwrite(fil, image) # write to location with same name
