import numpy as np
import os
import imgaug as ia
import imgaug.augmenters as iaa

import imgaug
import skimage.io

# Augmenters that are safe to apply to masks
# Some, such as Affine, have settings that make them unsafe, so always
# test your augmentation on masks
MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                   "Fliplr", "Flipud", "CropAndPad",
                   "Affine", "PiecewiseAffine"]

read_dir = "Mix_crypts/train/"
aug_string = "orginin"

def hook(images, augmenter, parents, default):
    """Determines which augmenters to apply to masks."""
    return augmenter.__class__.__name__ in MASK_AUGMENTERS

def load_mask(folder_path, image_id):
    """Generate instance masks for shapes of the given image ID.
    """
    print(image_id)
    mask_paths = os.path.join(folder_path, 'Annotation', os.path.splitext(image_id)[0]) + '_*'
    gland_path = os.path.join(folder_path, 'Annotation_gland', os.path.splitext(image_id)[0]) + '_*'
    #os.path.join(self.folder_path, 'Annotation', image_path)

    mask = skimage.io.imread_collection(mask_paths).concatenate()
    try:
        gland_mask = skimage.io.imread_collection(gland_path).concatenate()
        class_ids = np.array([1]* mask.shape[0] + [2] * gland_mask.shape[0])
        mask = np.concatenate((mask, gland_mask))
    except:
        class_ids = np.array([1] * mask.shape[0])
    mask = np.rollaxis(mask,0,3)
    mask = np.clip(mask,0,1)
    return mask, class_ids.astype(np.int32)


# Store shapes before augmentation to compare
ROOT_DIR = os.getcwd()


## Change this dir to the stage 1 training data
train_dir = os.path.join(ROOT_DIR, 'dataset/' + read_dir)

# Get train IDs
train_ids = next(os.walk(train_dir + '/Images'))[2]

try:
    os.makedirs("dataset/aug_images/" + read_dir + "Images/", )
except:
    pass
try:
    os.makedirs("dataset/aug_images/" + read_dir + "Annotation/")
except:
    pass
try:
    os.makedirs("dataset/aug_images/" + read_dir + "Annotation_gland/")
except:
    pass
for train_id in train_ids:
    image_path = os.path.join(train_dir, 'Images', train_id)
    image = skimage.io.imread(image_path)
    image = image[:, :, :3]

    mask, class_ids = load_mask(train_dir, train_id)

    image_shape = image.shape
    mask_shape = mask.shape
    # Make augmenters deterministic to apply similarly to images and masks
    if image.shape[0] + image.shape[1] < 4000:
        continue
        augmentation = iaa.Sequential([
            # iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
            # iaa.Fliplr(0.5),
            # iaa.Flipud(0.5),
            # iaa.Sometimes(0.5, iaa.AddElementwise((-10, 10), per_channel=0.5)),
            # iaa.Affine(rotate=(-100,100), scale=(.8, 1), translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
            # iaa.PadToFixedSize(width=1024, height=800),
            # iaa.CropToFixedSize(width=1024, height=800),
            # iaa.Sometimes(0.5, iaa.CropAndPad(percent=(-0.25, 0.25))),
        ], random_order=False)
    else:
        augmentation = iaa.Sequential([
            # iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
            iaa.Fliplr(0.5),
            # iaa.Flipud(0.5),
            # iaa.Sometimes(0.5, iaa.AddElementwise((-10, 10), per_channel=0.5)),
            iaa.Affine(scale=(1, 1.5)),
            iaa.PadToFixedSize(width=2000, height=1500),
            iaa.CropToFixedSize(width=1500, height=1000),
            # iaa.CropAndPad(keep_size=False, percent=(-0.5, 0))
            # iaa.Sometimes(0.5, iaa.CropAndPad(percent=(-0.25, 0.25))),
            # iaa.Sometimes(0.5, iaa.AddElementwise((-10, 10), per_channel=0.5)),
        ], random_order=False)
    det = augmentation.to_deterministic()
    image = det.augment_image(image)
    # Change mask to np.uint8 because imgaug doesn't support np.bool
    mask = det.augment_image(mask.astype(np.uint8),
                             hooks=imgaug.HooksImages(activator=hook))
    # Verify that shapes didn't change
    # assert image.shape == image_shape, "Augmentation shouldn't change image size"
    # assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
    # Change mask back to bool
    # mask = mask.astype(np.bool)

    skimage.io.imsave("dataset/aug_images/" + read_dir + "Images/" + os.path.splitext(train_id)[0] + aug_string + '.png', image)
    # print(image.shape)
    for i in range(mask_shape[2]):
        if class_ids[i] == 1:
            skimage.io.imsave("dataset/aug_images/" + read_dir + "Annotation/" + os.path.splitext(train_id)[0] + aug_string + '_' + str(i) + '.png', mask[:, :, i]*255)
        else:
            skimage.io.imsave("dataset/aug_images/" + read_dir + "Annotation_gland/" + os.path.splitext(train_id)[0]+ aug_string  + '_' + str(i) + '.png',
                              mask[:, :, i] * 255)
