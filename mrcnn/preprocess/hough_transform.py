import imgaug.augmenters as iaa
import skimage.io
import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import glob
import argparse

def get_hough_angle(img):
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    plt.imshow(imgray,cmap='gray')
    plt.show()
    imgray = cv.bilateralFilter(imgray, 10, 10, 10)
    plt.imshow(imgray,cmap='gray')
    plt.show()
    v = np.median(imgray)
    sigma = 0.4
    lower_thresh = int(max(0, (1.0 - sigma) * v))
    upper_thresh = int(min(255, (1.0 + sigma) * v))

    edges = cv.Canny(imgray, lower_thresh, upper_thresh)
    lines = cv.HoughLinesP(edges, 1, math.pi / 128, 40, None, 60, 10)
    if lines is None or len(lines) < 5:
        lines = cv.HoughLinesP(edges, 1, math.pi / 128, 40, None, 80, 20)
        if lines is None or len(lines) < 5:
            lines = cv.HoughLinesP(edges, 1, math.pi / 128, 20, None, 80, 20)
    plt.imshow(edges)
    plt.show()
    # lines = cv.HoughLines(edges, 1, math.pi/64 , 100, None, 80, 10)
    if lines is None:
        return 0
    # lines_theta = [line[0][1] for line in lines[:100]]
    lines_theta = np.array([np.arctan((line[0][3] - line[0][1]) / (line[0][2] - line[0][0]+.00001)) for line in lines[:100]])

    def reject_outliers(data, m=2):
        return data[abs(data - np.mean(data)) < m * np.std(data)]
    # print(len(lines_theta))
    # print(lines_theta)
    filter_lines_theta = reject_outliers(lines_theta, .5)
    if len(filter_lines_theta) == 0:
        filter_lines_theta = reject_outliers(lines_theta, 1)

    for l in lines[:100]:
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
        for line in l:
            pt1 = (line[0], line[1])
            pt2 = (line[2], line[3])
            cv.line(img, pt1, pt2, (0, 0, 255), 3)
    plt.imshow(img)
    plt.show()
    res = -np.mean(filter_lines_theta)
    if np.isnan(res):
        res = 0
    return res


def cut_black_area(img):
    mask = img > 0
    mask = mask.all(2)
    coords = np.argwhere(mask)
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1
    return x0, x1, y0, y1

def rotate_image(img, theta):
    augmentation = iaa.Sequential([
        iaa.Rotate((theta * 180 / math.pi), fit_output=True)
        # iaa.KeepSizeByResize(iaa.Rotate((-np.mean(lines_theta) * 180 / math.pi), fit_output=True))
    ], random_order=False)
    img = augmentation.augment_image(img)
    # plt.imshow(img)
    # plt.show()
    return img

# def save_hough_transform(image_path, dst_folder, label_folder=None):
#     img = cv.imread(image_path, cv.IMREAD_UNCHANGED)
#     x0, x1, y0, y1 = cut_black_area(img)
#     img = img[x0:x1, y0:y1]
#     theta = get_hough_angle(img)
#     img = skimage.io.imread(image_path)
#     img = rotate_image(img, theta)
#     skimage.io.imsave(dst_folder + os.path.splitext(os.path.basename(image_path))[0]+ '.png', img)
#     if label_folder:
#         mask = skimage.io.imread(label_folder + os.path.splitext(os.path.basename(image_path))[0] + ".png")
#         mask = rotate_image(mask, theta)
#         skimage.io.imsave(dst_folder + os.path.splitext(os.path.basename(image_path))[0]+ '.png', mask)

def save_hough_transform(image_path, output_path, label_folder=None):
    img = cv.imread(image_path, cv.IMREAD_UNCHANGED)
    x0, x1, y0, y1 = cut_black_area(img)
    img = img[x0:x1, y0:y1]
    theta = get_hough_angle(img)
    img = skimage.io.imread(image_path)
    img = rotate_image(img, theta)
    skimage.io.imsave(output_path, img)
    if label_folder:
        mask = skimage.io.imread(label_folder + os.path.splitext(os.path.basename(image_path))[0] + ".png")
        mask = rotate_image(mask, theta)
        skimage.io.imsave(dst_folder + os.path.splitext(os.path.basename(image_path))[0]+ '.png', mask)


def save_hough_transform_annotation(image_path, dst_folder, label_folder):
    img = cv.imread(image_path, cv.IMREAD_UNCHANGED)
    x0, x1, y0, y1 = cut_black_area(img)
    img = img[x0:x1, y0:y1]
    theta = get_hough_angle(img)
    img = skimage.io.imread(image_path)
    img = rotate_image(img, theta)
    skimage.io.imsave(dst_folder + "/Images/" + os.path.splitext(os.path.basename(image_path))[0]+ '.png', img)

    for i, mask_folder in enumerate(['/Annotation_gland/', '/Annotation/']):
        mask_names = glob.glob(label_folder + mask_folder + os.path.splitext(os.path.basename(image_path))[0] + '_*.png')
        for mask_name in mask_names:
            mask = skimage.io.imread(mask_name)
            mask = rotate_image(mask, theta)
            skimage.io.imsave(dst_folder + mask_folder + os.path.splitext(os.path.basename(mask_name))[0]+ '.png', mask)



if __name__ == '__main__':

    ## SET UP CONFIGURATION
    parser = argparse.ArgumentParser("hough_transform.py")
    parser.add_argument("--dest", help="Root folder path of the dest images Exp: images/", type=str,
                        required=True)
    parser.add_argument("--src",
                        help="Root folder path of the source images (will only process png files) Exp: if you have images in images/data1/1.png  images/data2/3.png you should pass: images/",
                        type=str, required=True)
    args = parser.parse_args()
    # for test_folder in ["test", "test_external"]:
    # for test_folder in os.listdir(args.src):
    src_folder = args.src
    dst_folder = args.dest

    search_pattern = os.path.join(src_folder, '**', '*.png')
    file_paths = glob.glob(search_pattern, recursive=True)
    print(f"Found {len(file_paths)} .png files to process.")
    for file_path in file_paths:
        print(f"Processing: {file_path}")

        # --- NEW LOGIC TO MIRROR DIRECTORY STRUCTURE ---

        # 1. Get the relative path from the source root.
        #    Ex: 'batch_A/image1.png'
        relative_path = os.path.relpath(file_path, src_folder)

        # 2. Change the extension if needed (the original code saves as .png).
        relative_path_png = os.path.splitext(relative_path)[0] + '.png'

        # 3. Create the full destination path.
        #    Ex: 'path/to/dest/batch_A/image1.png'
        output_path = os.path.join(dst_folder, relative_path_png)

        # 4. Create the destination directory if it doesn't exist.
        #    Ex: 'path/to/dest/batch_A/'
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # 5. Call the function with the full output path.
        save_hough_transform(file_path, output_path)







    # previous code
    # file_paths = glob.glob(src_folder + "" + '*.png')
    # # try:
    # #     os.makedirs(dst_folder + "/Images/", )
    # #     os.makedirs(dst_folder + "/Annotation/", )
    # #     os.makedirs(dst_folder + "/Annotation_gland/", )
    # # except Exception as e:
    # #
    # #     print(e)
    # #     pass
    # try:
    #     os.makedirs(dst_folder)
    #     # os.makedirs(dst_folder + "/labels/", )
    # except:
    #     pass
    # for file_path in file_paths:
    #     save_hough_transform(file_path, dst_folder)
    #     # save_hough_transform_annotation(file_path, dst_folder, src_folder)