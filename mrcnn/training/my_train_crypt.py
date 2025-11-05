import tensorflow as tf
import random
import os
import sys
import time
from my_crypts_dataset import CryptsDataset, CryptsConfig
import model as modellib
from model import log
import numpy as np
from imgaug import augmenters as iaa
import time
import argparse

#######################################################################################
## SET UP CONFIGURATION
parser = argparse.ArgumentParser("my_train_crypt.py")
parser.add_argument("--dataset", help="path to the dataset, Exp: dataset/Normalized_Images", type=str, required=True)
parser.add_argument("--dest", help="name of the output model, Exp:final.h5", type=str, required=True)
parser.add_argument("--model", help="path to the model, Exp: logs/no_transfer/mask_rcnn_crypt_0060.h5", type=str, required=False)
args = parser.parse_args()

seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)
random.seed(seed)
bowl_config = CryptsConfig()
bowl_config.display()
#######################################################################################

# Root directory of the project
ROOT_DIR = os.getcwd()

##base_dir = "dataset/Reinhard"
base_dir = args.dataset
train_dir = os.path.join(ROOT_DIR, base_dir + '/train/Images')
print(train_dir)

# Get train IDs
train_ids = next(os.walk(train_dir))[2]

# Training dataset
dataset_train = CryptsDataset()
dataset_train.load_bowl(train_ids, base_dir + '/train')
dataset_train.prepare()

# # Validation dataset, same as training.. will use pad64 on this one
val_dir = os.path.join(ROOT_DIR, base_dir + '/valid/Images/')
valid_ids = next(os.walk(val_dir))[2]
dataset_val = CryptsDataset()
dataset_val.load_bowl(valid_ids, base_dir + '/valid')
dataset_val.prepare()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

#COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

model = modellib.MaskRCNN(mode="training", config=bowl_config,
                          model_dir=MODEL_DIR)

#model.load_weights(COCO_MODEL_PATH, by_name=True,
#                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
#                            "mrcnn_bbox", "mrcnn_mask"])

if args.model:
    model.load_weights(args.model, by_name=True)

start_time = time.time()

## Augment True will perform flipud fliplr and 90 degree rotations on the 512x512 images

# Image augmentation
# http://imgaug.readthedocs.io/en/latest/source/augmenters.html

# ## This should be the equivalente version of my augmentations using the imgaug library
# ## However, there are subtle differences so I keep my implementation
augmentation = iaa.Sequential([
    #iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.Sometimes(0.5,iaa.Affine(rotate=(-180,180))),
    iaa.Sometimes(0.5,iaa.CropAndPad(percent=(-0.25, 0.25))),
    iaa.Sometimes(0.5, iaa.AddElementwise((-10, 10), per_channel=0.5)),
], random_order=True)

def _load_augmentation_aug_all():
    """ Load image augmentation model """

    def sometimes(aug):
        return iaa.Sometimes(0.5, aug)

    return iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            iaa.Flipud(0.5),  # vertically flip 20% of all images
            # crop images by -5% to 10% of their height/width
            sometimes(iaa.CropAndPad(
                percent=(-0.05, 0.1),
                pad_mode='constant',
                pad_cval=(0, 255)
            )),
            sometimes(iaa.Affine(
                # scale images to 80-120% of their size, individually per axis
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                # translate by -20 to +20 percent (per axis)
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-45, 45),  # rotate by -45 to +45 degrees
                shear=(-16, 16),  # shear by -16 to +16 degrees
                # use nearest neighbour or bilinear interpolation (fast)
                order=[0, 1],
                # if mode is constant, use a cval between 0 and 255
                cval=(0, 255),
                # use any of scikit-image's warping modes
                # (see 2nd image from the top for examples)
                mode='constant'
            )),
            # execute 0 to 5 of the following (less important) augmenters per
            # image don't execute all of them, as that would often be way too
            # strong
            iaa.SomeOf((0, 5),
                       [
                # convert images into their superpixel representation
                sometimes(iaa.Superpixels(
                    p_replace=(0, 1.0), n_segments=(20, 200))),
                iaa.OneOf([
                    # blur images with a sigma between 0 and 3.0
                    iaa.GaussianBlur((0, 3.0)),
                    # blur image using local means with kernel sizes
                    # between 2 and 7
                    iaa.AverageBlur(k=(2, 7)),
                    # blur image using local medians with kernel sizes
                    # between 2 and 7
                    iaa.MedianBlur(k=(3, 11)),
                ]),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(
                            0.75, 1.5)),  # sharpen images
                iaa.Emboss(alpha=(0, 1.0), strength=(
                    0, 2.0)),  # emboss images
                # search either for all edges or for directed edges,
                # blend the result with the original image using a blobby mask
                iaa.SimplexNoiseAlpha(iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0.5, 1.0)),
                    iaa.DirectedEdgeDetect(
                        alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                ])),
                # add gaussian noise to images
                iaa.AdditiveGaussianNoise(loc=0, scale=(
                    0.0, 0.05*255), per_channel=0.5),
                iaa.OneOf([
                    # randomly remove up to 10% of the pixels
                    iaa.Dropout((0.01, 0.1), per_channel=0.5),
                    iaa.CoarseDropout((0.03, 0.15), size_percent=(
                        0.02, 0.05), per_channel=0.2),
                ]),
                # invert color channels
                iaa.Invert(0.05, per_channel=True),
                # change brightness of images (by -10 to 10 of original value)
                iaa.Add((-10, 10), per_channel=0.5),
                # change hue and saturation
                iaa.AddToHueAndSaturation((-20, 20)),
                # either change the brightness of the whole image (sometimes
                # per channel) or change the brightness of subareas
                iaa.OneOf([
                    iaa.Multiply(
                                (0.5, 1.5), per_channel=0.5),
                    iaa.FrequencyNoiseAlpha(
                        exponent=(-4, 0),
                        first=iaa.Multiply(
                            (0.5, 1.5), per_channel=True),
                        second=iaa.ContrastNormalization(
                            (0.5, 2.0))
                    )
                ]),
                # improve or worsen the contrast
                iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
                iaa.Grayscale(alpha=(0.0, 1.0)),
                # move pixels locally around (with random strengths)
                sometimes(iaa.ElasticTransformation(
                    alpha=(0.5, 3.5), sigma=0.25)),
                # sometimes move parts of the image around
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
            ],
                random_order=True
            )
        ],
        random_order=True
    )

# augmentation = False

model.train(dataset_train, dataset_val,
           learning_rate=bowl_config.LEARNING_RATE,
           epochs=1,
            augmentation=augmentation,
            #augment=True,
           layers="heads")

model.train(dataset_train, dataset_val,
           learning_rate=bowl_config.LEARNING_RATE,
           epochs=20,
           layers="4+")
augmentation = _load_augmentation_aug_all()
model.train(dataset_train, dataset_val,
            learning_rate=bowl_config.LEARNING_RATE,
            epochs=30,
            augmentation=augmentation,
            # augment=True,
            layers="all")

end_time = time.time()
ellapsed_time = (end_time - start_time) / 3600

print(model.log_dir)
model_path = os.path.join(model.log_dir, args.dest)
model.keras_model.save_weights(model_path)
