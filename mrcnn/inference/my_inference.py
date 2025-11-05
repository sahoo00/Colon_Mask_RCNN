seed=123
from keras import backend as K
import numpy as np
np.random.seed(seed)
import tensorflow as tf
tf.set_random_seed(seed)
import random
random.seed(seed)
import skimage.io
from skimage import img_as_ubyte
import model as modellib
import pandas as pd
import os
import my_functions as f
import imgaug.augmenters as iaa
from mrcnn.model import log
from visualize import display_instances
import matplotlib.pyplot as plt
import cv2 as cv
import math
import argparse
import time
import glob

#######################################################################################
## SET UP CONFIGURATION
parser = argparse.ArgumentParser("my_inference.py")
parser.add_argument("--src", help="path to the src, Exp: dataset/Normalized_Images/", type=str, required=True)
parser.add_argument("--dest", help="path to the dest, Exp: dataset/Normalized_Images/", type=str, required=True)
parser.add_argument("--model", help="name of the model, Exp:final.h5", type=str, required=True)
args = parser.parse_args()



#######################################################################################
## SET UP CONFIGURATION
from config import Config

class BowlConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "Inference"
    BACKBONE = "resnet50"
    IMAGE_RESIZE_MODE = "square" ## tried to modfied but I am using other git clone
    ## No augmentati
    ZOOM = False
    ASPECT_RATIO = 1
    MIN_ENLARGE = 1
    IMAGE_MIN_SCALE = False ## Not using this

    # IMAGE_MIN_DIM = 512 # We scale small images up so that smallest side is 512
    # IMAGE_MAX_DIM = False
    IMAGE_MIN_DIM = 768
    IMAGE_MAX_DIM = 1024


    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    DETECTION_MAX_INSTANCES = 512
    DETECTION_NMS_THRESHOLD =  0.4
    DETECTION_MIN_CONFIDENCE = 0.7

    LEARNING_RATE = 0.0001
    
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1 + 1 # background + nuclei

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)  # anchor side in pixels
    # RPN_ANCHOR_SCALES = (128, 256, 512)

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 200

    USE_MINI_MASK = True


inference_config = BowlConfig()
inference_config.display()
#######################################################################################


ROOT_DIR = os.getcwd()
#MODEL_DIR = os.path.join(ROOT_DIR, "logs")
MODEL_DIR = ROOT_DIR

## Change this with the path to the last epoch of train

model_path = args.model

## change this with the correct paths for images and sample submission
src_path = args.src
dest_path = args.dest
test_path = os.path.join(ROOT_DIR, src_path)
# sample_submission = pd.read_csv('dataset/Normalized_Images/test.txt', delimiter="\t")

# sample_submission = pd.DataFrame({"ImageId": ["50HD0147.png"]})
image_ids = [image_id for image_id in os.listdir(src_path) if image_id.endswith(".png")]
sample_submission = pd.DataFrame({"ImageId": image_ids})

print("Loading weights from ", model_path)

try:
    os.mkdir(dest_path)
except:
    pass
try:
    os.mkdir(os.path.join(dest_path, "predicted_mask/"))
    os.mkdir(os.path.join(dest_path, "predicted_images"))
except:
    pass
import time
start_time = time.time()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)
model.load_weights(MODEL_DIR + '/' +  model_path, by_name=True)


ImageId_d = []
EncodedPixels_d = []

n_images= len(sample_submission.ImageId)

def fit_hough_rotate(image_path):
    img = cv.imread(image_path, cv.IMREAD_UNCHANGED)
    # img = original
    mask = img > 0
    mask = mask.all(2)
    coords = np.argwhere(mask)
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1
    img = img[x0:x1, y0:y1]
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgray = cv.bilateralFilter(imgray, 10, 50, 50)
    v = np.median(imgray)
    sigma = 0.33
    lower_thresh = int(max(0, (1.0 - sigma) * v))
    upper_thresh = int(min(255, (1.0 + sigma) * v))

    edges = cv.Canny(imgray, lower_thresh, upper_thresh)
    lines = cv.HoughLinesP(edges, 1, math.pi / 128, 40, None, 60, 10)
    if lines is None or len(lines) < 4:
        lines = cv.HoughLinesP(edges, 1, math.pi / 128, 40, None, 80, 20)
        if lines is None or len(lines) < 4:
            lines = cv.HoughLinesP(edges, 1, math.pi / 128, 20, None, 80, 20)

    # lines = cv.HoughLines(edges, 1, math.pi/64 , 100, None, 80, 10)

    # lines_theta = [line[0][1] for line in lines[:100]]
    lines_theta = np.array([np.arctan((line[0][3] - line[0][1]) / (line[0][2] - line[0][0])) for line in lines[:100]])

    def reject_outliers(data, m=2):
        return data[abs(data - np.mean(data)) < m * np.std(data)]

    lines_theta = reject_outliers(lines_theta, 1)

    # for l in lines[:100]:
    # for rho, theta in l:
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     x0 = a * rho
    #     y0 = b * rho
    #     x1 = int(x0 + 1000 * (-b))
    #     y1 = int(y0 + 1000 * (a))
    #     x2 = int(x0 - 1000 * (-b))
    #     y2 = int(y0 - 1000 * (a))
    #     cv.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    # for line in l:
    #     pt1 = (line[0], line[1])
    #     pt2 = (line[2], line[3])
    #     cv.line(img, pt1, pt2, (0, 0, 255), 3)

    augmentation = iaa.Sequential([
        iaa.Rotate((-np.mean(lines_theta) * 180 / math.pi), fit_output=True)
        # iaa.KeepSizeByResize(iaa.Rotate((-np.mean(lines_theta) * 180 / math.pi), fit_output=True))
    ], random_order=False)
    img = augmentation.augment_image(skimage.io.imread(image_path))
    plt.imshow(img)
    plt.show()
    return img

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

for i in np.arange(n_images):
    image_id = sample_submission.ImageId[i]
    print('Start detect',i, '  ' ,image_id)
    ##Set seeds for each image, just in case..
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    ## Load the image
    image_path = os.path.join(test_path, image_id)
    original_image = skimage.io.imread(image_path)
    # original_image = cv.imread(image_path, cv.IMREAD_UNCHANGED)
    # original_image = fit_hough_rotate(image_path)
    ####################################################################
    ## This is needed for tki8,ki8h,kii8,ki8i8,ki8,ki88,ki8,ki8e,ki8,ki,ki8888,ki8ki8ki8,ki8 ,ki8,ki8stage 2 image that has only one channel
    if len(original_image.shape)<3:
        original_image = img_as_ubyte(original_image)
        original_image = np.expand_dims(original_image,2)
        original_image = original_image[:,:,[0,0,0]] # flip r and b
    ####################################################################
    original_image = original_image[:,:,:3]
    augmentation = iaa.Sequential([
        iaa.PadToAspectRatio(1.5, position="center"),
        iaa.Resize({"height": 768, "width": 1024}),
        # iaa.CropAndPad(percent=(-0.3))
        # iaa.Rotate((-30))
        # iaa.Fliplr(1),
        # iaa.Flipud(1),
        ], random_order=False)
    det = augmentation.to_deterministic()
    original_image = augmentation.augment_image(original_image)
    ## Make prediction for that image
    results = model.detect([original_image], verbose=1)

    # Display results
    ax = get_ax(1)
    r = results[0]
    display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                                ['bg', 'crypt', 'gland'], r['scores'],
                                title=image_id, dest=os.path.join(dest_path, "predicted_images/"))

    ## Proccess prediction into rle
    pred_masks = results[0]['masks']
    scores_masks = results[0]['scores']
    class_ids = results[0]['class_ids']

    pred = np.zeros(original_image.shape[:2])
    for i in range(1, inference_config.NUM_CLASSES):
        if sum(class_ids == (inference_config.NUM_CLASSES-i)): # using inference_config.NUM_CLASSES-i instead of i to fix the mismatch in semantic dataset
            mask = np.any(pred_masks[:,:,class_ids==(inference_config.NUM_CLASSES-i)], 2)
            pred[mask] = i

    for i, class_id in enumerate(class_ids):
        if class_id == 1:  # using inference_config.NUM_CLASSES-i instead of i to fix the mismatch in semantic dataset
            cv.imwrite(os.path.join(dest_path, "predicted_mask/", image_id[:-len(".png")]+ "_" + str(i) + ".png"), pred_masks[:, :, i]*255)

    # cv2.imwrite(os.path.join("prediction", image_id), pred)
    # plt.imsave(os.path.join("prediction", image_id), pred)
    # np.savetxt(os.path.join("benchmark_correct_size/MaskRCNN_square_rotated/test", os.path.basename(image_id))+".txt", pred, fmt='%d')

# f.write2csv('submission.csv', ImageId_d, EncodedPixels_d)

end_time = time.time()
ellapsed_time = (end_time-start_time)/3600
print('Time required to train ', ellapsed_time, 'hours')

# python3 my_inference.py --src dataset/Whole_Slides_Segments_Processed/ht/Boiling-2min/S.19.4704/ --dest dataset/Whole_Slides_Segments_Processed/inf/Boiling-2min/S.19.4704/ --model=notransfer.h5