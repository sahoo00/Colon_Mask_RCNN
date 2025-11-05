from mrcnn.utils import Dataset
import os
import numpy as np
import skimage.io
import glob
import cv2

from mrcnn.config import Config


class CryptsConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "crypt"
    BACKBONE = "resnet50"
    WEIGHT_DECAY = 0.000001

    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 768
    IMAGE_MAX_DIM = 1024
    IMAGE_MIN_SCALE = False  ## Not using this


    IMAGE_CHANNEL_COUNT = 3
    # Augmentation parameters
    ASPECT_RATIO = 1  ## Maximum aspect ratio modification when scaling
    MIN_ENLARGE = 1 ## Minimum enlarging of images, note that this will be randomized
    ZOOM = 1.5  ## Maximum zoom per image, note that this will be randomized


    ROT_RANGE = 10.
    CHANNEL_SHIFT_RANGE = 15

    LEARNING_RATE = 0.001
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 4
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2   # background + u-shape + circular glands


    # Use smaller anchors because our image and objects are small
    # RPN_ANCHOR_SCALES = (128, 256, 512)  # anchor side in pixels
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)  # anchor side in pixels
    #RPN_ANCHOR_RATIOS = [0.5, 1, 1.5 , 2, 2.5]

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 512

    STEPS_PER_EPOCH = 600 // IMAGES_PER_GPU
    VALIDATION_STEPS = 50  # 2//IMAGES_PER_GPU ## We are training with the whole dataset so validation is not very meaningfull, I put a two here so it is faster. We either use train loss or calculate in a separate procceses the mAP for each epoch

    # use small validation steps since the epoch is small
    # VALIDATION_STEPS = 5

    USE_MINI_MASK = True

    MAX_GT_INSTANCES = 256

    DETECTION_MAX_INSTANCES = 512



class CryptsDataset(Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load_bowl(self, folderpaths, folderpath):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("bowl", 1, "crypt")
        self.add_class("bowl", 2, "gland")

        #Image size must be dividable by 2 at least 6 times to avoid fractions when downscaling and upscaling.For example, use 256, 320, 384, 448, 512, ... etc. 

        self.folder_path = folderpath
        # Add images
        for i in range(len(folderpaths)):
            self.add_image("bowl", image_id=i, path=folderpaths[i])

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        image_path = info['path']
        image_path = os.path.join(self.folder_path, 'Images', image_path)
        image = skimage.io.imread(image_path)
        #image = cv2.imread(image_path)
        image = image[:,:,:3]
        return image
        
    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "bowl":
            return info
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        image_path = info['path']
        mask_paths = os.path.join(self.folder_path, 'Annotation', os.path.splitext(image_path)[0])+ '_*'
        gland_path = os.path.join(self.folder_path, 'Annotation_gland', os.path.splitext(image_path)[0])+ '_*'
        #os.path.join(self.folder_path, 'Annotation', image_path)
        class_ids = np.array([])
        mask=np.array([])
        try:
            mask = skimage.io.imread_collection(mask_paths).concatenate()
            class_ids = np.array([1] * mask.shape[0])
        except:
            pass
        try:
            gland_mask = skimage.io.imread_collection(gland_path).concatenate()
            class_ids = np.concatenate((class_ids, [2] * gland_mask.shape[0]))
            if mask != np.array([]):
                mask = np.concatenate((mask, gland_mask))
            else:
                mask = gland_mask
        except:
            class_ids = np.array([1] * mask.shape[0])
        mask = np.rollaxis(mask,0,3)
        mask = np.clip(mask,0,1)
        return mask, class_ids.astype(np.int32)




