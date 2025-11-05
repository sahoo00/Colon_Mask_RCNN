#!/usr/bin/env python
# coding: utf-8

# In[1]:


## README: 
# input: a folder contains the images you want to rotate, 
#            and the name of a folder to hold output (doesn't exist before)
# output: a folder with the set name, contains the roateted images 
# just need to call:       iterate_dir_save(images_folder, des_folder)


# In[2]:
import pdb
import cv2
import os
from matplotlib import pyplot as plt
import numpy as np
# for rotation
import argparse
import imutils
import glob
import imgaug.augmenters as iaa



# In[3]:


#step 1: find an ellipse to fit the contour of the u shape
# input: original image, RGB valued
# output: contour, ellipse-> the first ellipse, gray -> grayscaled image
def find_ellipse(image):

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #gray = cv2.cvtColor(cv2.UMat(image), cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    
    # find contours
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    # find the biggest contour
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    cnt = max(contour_sizes, key=lambda x: x[0])[1]
    #cnt = contours[0]
    
    #contours,hierarchy = cv2.findContours(thresh, 1, 2)
    #cnt = contours[0]
    #cv2.drawContours(image,cnt, -1, 100, -1)
    #ellipse = cv2.fitEllipse(cnt)
    while True:
        try:
            ellipse = cv2.fitEllipse(cnt)
            break
        except:
            flag = 1
            print("Oops!  This gland is invalid or circular")
            return cnt, ([0,0],[0,0]), gray
            break
    
    #cv2.ellipse(gray,ellipse,(0,100,0),50)
    
    return cnt,ellipse, gray


# In[4]:


# step2: rotate the image by the angle of the ellipse
# make the image horizontal

def rotate1(image,elps):
    #cnts,elps, gray_elips, flag = find_ellipse(image)
    #print(elps[2])
    #rotated = imutils.rotate_bound(image, elps[2])
    # flag herer helps to find if the we can fit ellipse 
    # if yes, then 0; otherwise 1; initialize as 0
    
    # plt.imshow(image),plt.show()
    flag = 0
    
    angle  = elps[2]
    print(angle)
    if angle > 90:
        rotated = imutils.rotate_bound(image, 270-angle)
    elif angle == 0:
        flag = 1
        rotated = image
    else:    
        rotated = imutils.rotate_bound(image, 90-angle)
    
    rotated_gray = cv2.cvtColor(rotated,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(rotated_gray, 127, 255, 0)
    contours,hierarchy = cv2.findContours(thresh, 1, 2)
    #print(contours)
    #cnt = contours[0]
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    cnt = max(contour_sizes, key=lambda x: x[0])[1]
    #cv2.drawContours(image,cnt, -1, 100, -1)
    while True:
        try:
            ellipse2 = cv2.fitEllipse(cnt)
            break
        except:
            flag = 1
            print('here flag2')
            print(flag)
            print("Oops!  This gland is invalid")
            return rotated, rotated_gray, ([0,0],[0,0]), flag
    
    cv2.ellipse(rotated_gray,ellipse2,(255,255,255),10)
    # plt.imshow(rotated_gray),plt.show()
    
    return rotated, rotated_gray, ellipse2, flag


# In[5]:


# input: rotated image, and the second ellipse
# output: direction 
# connected components

def direc_det(image):
    
    cnt,elps, gray_elips = find_ellipse(image)
    rotated, rotated_gray, ellipse2, flag = rotate1(image, elps)
    gray = cv2.cvtColor(rotated,cv2.COLOR_BGR2GRAY)    
    #plt.imshow(image),plt.show()
    #plt.imshow(rotated_gray),plt.show()
    
    print('flag = ', flag)
    
    #flag = 1
    # flag = 0
    if flag == 0:
        # thresh = ellipse2[0][0]
        # if ellipse2[0][0] < ellipse2[0][1]:
        #     thresh = ellipse2[0][0]
        # else:
        #     thresh = ellipse2[0][1]
        left_ori = gray[:,:int(ellipse2[0][0])]
        right_ori = gray[:,int(ellipse2[0][0]):]

        #plt.imshow(left_ori),plt.show()
        #plt.imshow(right_ori),plt.show()

        ret1, thresh1 = cv2.threshold(left_ori,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        ret2, thresh2 = cv2.threshold(right_ori,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        nlabels_left, labels_left, stats_left, centroids_left = cv2.connectedComponentsWithStats(thresh1)
        nlabels_right, labels_r, stats_r, centroids_r = cv2.connectedComponentsWithStats(thresh2)

        if nlabels_left > nlabels_right:
            direction = 'left'
            print(direction)
            print('left = ', nlabels_left)
            print('right = ', nlabels_right)
            
        elif nlabels_right >=nlabels_left:
            direction = 'right'
            print(direction)
            print('left = ', nlabels_left)
            print('right = ', nlabels_right)
    
    # flag = 1
    elif flag == 1:
        left_ori = gray[:,:int(len(gray/2))]
        right_ori = gray[:,int(len(gray/2)):]

        #plt.imshow(left_ori),plt.show()
        #plt.imshow(right_ori),plt.show()

        ret1, thresh1 = cv2.threshold(left_ori,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        ret2, thresh2 = cv2.threshold(right_ori,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        nlabels_left, labels_left, stats_left, centroids_left = cv2.connectedComponentsWithStats(thresh1)
        nlabels_right, labels_r, stats_r, centroids_r = cv2.connectedComponentsWithStats(thresh2)
        
        if nlabels_left < nlabels_right:
            direction = 'left'
            print(direction)
            print('left = ', nlabels_left)
            print('right = ', nlabels_right)
            
        elif nlabels_left >= nlabels_right:
            direction = 'right'
        
            print(direction)
            print('left = ', nlabels_left)
            print('right = ', nlabels_right)
            
    return rotated, elps, direction


# In[6]:


# input: img: rotated rgb image
# output: final rotated image
def rotate2(img, direction):
    if direction == 'left':
        rotated = imutils.rotate_bound(img, -90)
    else:
        rotated = imutils.rotate_bound(img, 90)
        
    # plt.imshow(rotated),plt.show()
    return rotated


# In[1]:

def fit_box(original):
    mask = original > 0
    mask = mask.all(2)
    coords = np.argwhere(mask)
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1
    return original[x0:x1, y0:y1]

def iterate_dir_save(mask_directory, original_directory, des_dir):
    
    # create a dest_directory
    try:
        os.mkdir(des_dir)
    except:
        pass
    print(original_directory)
    for rgb_img_path in os.listdir(original_directory):
        mask_names = glob.glob(os.path.join(mask_directory, os.path.splitext(os.path.basename(rgb_img_path))[0] + '_*.png'))
        for mask_path in mask_names:
            if mask_path.endswith(".png"):
                try:
                    print(mask_path)
                    #pdb.set_trace()
                    curr_img = cv2.imread(mask_path)
                    curr_img_height, curr_img_width = curr_img.shape[:2]
                    #pdb.set_trace()
                    # rgb_img_filename=mask_path[:mask_path.rfind('_')] + ".png"
                    # rgb_img_path = original_directory + mask_path[:mask_path.rfind('_')] + ".png"
                    #rgb_img = cv2.resize(cv2.imread(original_directory + rgb_img_path), (curr_img_width, curr_img_height))
                    rgb_img = cv2.resize(cv2.imread(original_directory + rgb_img_path), (curr_img_width, curr_img_height))
                    augmentation = iaa.Sequential([
                        iaa.PadToAspectRatio(1.5, position="center"),
                        iaa.Resize({"height": 768, "width": 1024}),
                    ], random_order=False)
                    rgb_img = augmentation.augment_image(rgb_img)
                    rotated, elps, direction = direc_det(curr_img)
                    rgb_img, _, _, _ = rotate1(rgb_img, elps)
                    rotated2 = rotate2(rotated, direction)
                    rgb_img = rotate2(rgb_img, direction)
                    rotated2 = cv2.cvtColor(rotated2, cv2.COLOR_BGR2GRAY)
                    res = cv2.bitwise_and(rgb_img, rgb_img, mask=rotated2)
                    res = fit_box(res)
                except:
                    continue
                #path = '/Users/Flora/Desktop/Boolean_Lab_Research/dataset/new_crypts/train/edited_annotation'
                status = cv2.imwrite(os.path.join(des_dir, os.path.basename(mask_path)), res)
                print("Image written to file-system : ", status)
                continue
            else:
                continue




# def iterate_dir_save(mask_root_dir, image_root_dir, dest_root_dir):
#     """
#     Recursively finds masks, processes them with their corresponding images from a
#     different structure, and saves the output to a mirrored directory structure.
#     """
#     print(f"Starting process...")
#     print(f"Mask source: {mask_root_dir}")
#     print(f"Image source: {image_root_dir}")
#     print(f"Destination: {dest_root_dir}")

#     # Walk through the mask directory tree
#     for dirpath, _, filenames in os.walk(mask_root_dir):
#         # We only care about directories named 'predicted_mask'
#         if not dirpath.endswith("predicted_mask"):
#             continue

#         for mask_filename in filenames:
#             if not mask_filename.endswith(".png"):
#                 continue

#             print(f"\nProcessing mask: {mask_filename}")
            
#             try:
#                 # --- 1. Construct all necessary paths ---
#                 mask_path = os.path.join(dirpath, mask_filename)
                
#                 # Get the base name of the original image (e.g., 'image1.png' from 'image1_mask0.png')
#                 base_image_name = mask_filename.rsplit('_', 1)[0] + ".png"
                
#                 # --- MODIFIED LOGIC TO HANDLE MISMATCH ---
#                 # Get the relative path from the mask root (e.g., 'd1/predicted_mask')
#                 relative_mask_dir = os.path.relpath(dirpath, mask_root_dir)
                
#                 # Remove the '/predicted_mask' part to get the correct image and destination path (e.g., 'd1')
#                 image_dest_relative_dir = os.path.dirname(relative_mask_dir)

#                 # Construct the full path to the corresponding original image
#                 rgb_img_path = os.path.join(image_root_dir, image_dest_relative_dir, base_image_name)
                
#                 # Construct the full path for the output file
#                 output_path = os.path.join(dest_root_dir, image_dest_relative_dir, mask_filename)
                
#                 # --- 2. Check if corresponding image exists ---
#                 if not os.path.exists(rgb_img_path):
#                     print(f"  - Warning: Corresponding image not found at {rgb_img_path}. Skipping.")
#                     continue

#                 # --- 3. Run the processing pipeline (this part is unchanged) ---
#                 mask_img = cv2.imread(mask_path)
#                 if mask_img is None: continue
                
#                 curr_img_height, curr_img_width = mask_img.shape[:2]
                
#                 rgb_img = cv2.imread(rgb_img_path)
#                 if rgb_img is None: continue
                
#                 rgb_img = cv2.resize(rgb_img, (curr_img_width, curr_img_height))
                
#                 augmentation = iaa.Sequential([
#                     iaa.PadToAspectRatio(1.5, position="center"),
#                     iaa.Resize({"height": 768, "width": 1024}),
#                 ], random_order=False)
#                 rgb_img = augmentation.augment_image(rgb_img)

#                 rotated, elps, direction = direc_det(mask_img)
#                 rgb_img, _, _, _ = rotate1(rgb_img, elps)
#                 rotated2 = rotate2(rotated, direction)
#                 rgb_img = rotate2(rgb_img, direction)
#                 rotated2 = cv2.cvtColor(rotated2, cv2.COLOR_BGR2GRAY)
#                 res = cv2.bitwise_and(rgb_img, rgb_img, mask=rotated2)
#                 res = fit_box(res)
                
#                 # --- 4. Create output directory and save file ---
#                 output_dir = os.path.dirname(output_path)
#                 os.makedirs(output_dir, exist_ok=True)
                
#                 status = cv2.imwrite(output_path, res)
#                 if status:
#                     print(f"  - Success: Image written to {output_path}")
#                 else:
#                     print(f"  - Failed: Could not write image to {output_path}")

#             except Exception as e:
#                 print(f"  - An error occurred while processing {mask_filename}: {e}")
#                 continue
# In[38]:


parser = argparse.ArgumentParser("U-shape-bottom-pipeline.py")
parser.add_argument("--mask_src",
                    help="Folder path of the predicted mask from the inference part, Exp: dataset/test_masks/",
                    type=str, required=True)
parser.add_argument("--image_src",
                    help="Folder path of the images(the images you used for inference) Exp: dataset/test_images/",
                    type=str, required=True)
parser.add_argument("--dest",
                    help="Output folder path Exp: dataset/test_images/",
                    type=str, required=True)
args = parser.parse_args()
# just call iterate_dir_save
iterate_dir_save(args.mask_src, args.image_src, args.dest)


# # python U-shape-bottom-pipeline.py --mask_src dataset/Whole_Slides_Segments_Processed/inf/ --image_src dataset/Whole_Slides_Segments_Processed/ht/ --dest dataset/Whole_Slides_Segments_Processed/us/




