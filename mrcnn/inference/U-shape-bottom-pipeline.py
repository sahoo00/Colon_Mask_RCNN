import os, glob, argparse
import cv2
import numpy as np
import imutils
import imgaug.augmenters as iaa

# ---------------------------
# OPTION-1: fit ellipse on ALL foreground pixels
# ---------------------------

def find_ellipse(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pts = cv2.findNonZero(mask)
    if pts is None or len(pts) < 5:
        return np.zeros((0,1,2), dtype=np.int32), ([0,0],[0,0]), gray
    if len(pts) > 8000:
        idx = np.random.choice(len(pts), 8000, replace=False)
        pts = pts[idx]
    ellipse = cv2.fitEllipse(pts.astype(np.float32))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea) if contours else np.zeros((0,1,2), np.int32)
    return cnt, ellipse, gray

# ---------------------------
# rotate1: rotate by ellipse angle, then REFIT ellipse on ALL foreground
# ---------------------------

def rotate1(image, elps):
    flag = 0
    angle = elps[2]
    if angle > 90:
        rotated = imutils.rotate_bound(image, 270 - angle)
    elif angle == 0:
        flag = 1
        rotated = image
    else:
        rotated = imutils.rotate_bound(image, 90 - angle)

    rotated_gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    _, thr = cv2.threshold(rotated_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pts = cv2.findNonZero(thr)
    if pts is None or len(pts) < 5:
        flag = 1
        return rotated, rotated_gray, ([0,0],[0,0]), flag
    if len(pts) > 8000:
        idx = np.random.choice(len(pts), 8000, replace=False)
        pts = pts[idx]
    try:
        ellipse2 = cv2.fitEllipse(pts.astype(np.float32))
    except:
        flag = 1
        return rotated, rotated_gray, ([0,0],[0,0]), flag

    cv2.ellipse(rotated_gray, ellipse2, (255,255,255), 10)  # optional overlay
    return rotated, rotated_gray, ellipse2, flag

# ---------------------------
# direction detector (split by ellipse center or midline)
# ---------------------------

def direc_det(image):
    _, elps, _ = find_ellipse(image)
    rotated, rotated_gray, ellipse2, flag = rotate1(image, elps)
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)

    if flag == 0 and isinstance(ellipse2, tuple):
        cx = int(ellipse2[0][0])
    else:
        cx = gray.shape[1] // 2

    left_ori  = gray[:, :cx]
    right_ori = gray[:,  cx:]

    _, thresh1 = cv2.threshold(left_ori,  0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, thresh2 = cv2.threshold(right_ori, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    nlabels_left,  *_ = cv2.connectedComponentsWithStats(thresh1)
    nlabels_right, *_ = cv2.connectedComponentsWithStats(thresh2)

    direction = 'left' if nlabels_left > nlabels_right else 'right'
    return rotated, elps, direction

def rotate2(img, direction):
    return imutils.rotate_bound(img, -90 if direction == 'left' else 90)

def fit_box(original):
    mask = original > 0
    mask = mask.all(2)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return original
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1
    return original[x0:x1, y0:y1]

# ---------------------------
# Upright check
# ---------------------------

def opening_is_up(mask_bgr, top_frac=0.45, bottom_frac=0.45):
    gray = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    ys, xs = np.where(bw > 0)
    if len(ys) == 0:
        return True
    y0, y1 = ys.min(), ys.max()
    roi = bw[y0:y1+1, :]
    roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    row_counts = (roi // 255).sum(axis=1).astype(np.float32)
    k = 9 if len(row_counts) >= 9 else max(3, (len(row_counts)//2)*2 + 1)
    kernel = np.ones(k, dtype=np.float32) / k
    row_counts_s = np.convolve(row_counts, kernel, mode='same')
    if len(row_counts_s) == 0:
        return True
    ymin_idx = int(np.argmin(row_counts_s))
    h = len(row_counts_s)
    pos = ymin_idx / max(h - 1, 1)
    if pos <= top_frac:
        return True
    if pos >= 1.0 - bottom_frac:
        return False
    return True

def ensure_u_upright(mask_bgr, rgb_bgr):
    if not opening_is_up(mask_bgr):
        mask_bgr = cv2.rotate(mask_bgr, cv2.ROTATE_180)
        rgb_bgr  = cv2.rotate(rgb_bgr,  cv2.ROTATE_180)
    return mask_bgr, rgb_bgr

# ---------------------------
# Runner for one (mask_dir, image_dir) pair
# ---------------------------

def iterate_dir_save_accepted(mask_directory, original_directory, des_dir):
    os.makedirs(des_dir, exist_ok=True)
    print("[PROCESS] images:", original_directory)
    print("[PROCESS] masks :", mask_directory)
    print("[OUTPUT ] dest  :", des_dir)

    for rgb_fname in os.listdir(original_directory):
        base = os.path.splitext(os.path.basename(rgb_fname))[0]
        mask_names = glob.glob(os.path.join(mask_directory, base + '_*.png'))
        if not mask_names:
            continue

        rgb_full = os.path.join(original_directory, rgb_fname)
        rgb_src = cv2.imread(rgb_full, cv2.IMREAD_COLOR)
        if rgb_src is None:
            print("  ! Could not read RGB:", rgb_full)
            continue

        for mask_path in mask_names:
            if not mask_path.lower().endswith(".png"):
                continue
            try:
                mask_img = cv2.imread(mask_path, cv2.IMREAD_COLOR)
                if mask_img is None:
                    print("  ! Could not read mask:", mask_path)
                    continue

                mh, mw = mask_img.shape[:2]
                rgb_resize = cv2.resize(rgb_src, (mw, mh))
                augmentation = iaa.Sequential([
                    iaa.PadToAspectRatio(1.5, position="center"),
                    iaa.Resize({"height": 768, "width": 1024}),
                ], random_order=False)
                rgb_img_aug = augmentation.augment_image(rgb_resize)

        
                rotated_mask1, elps, direction = direc_det(mask_img)
                rgb_rot1, _, _, _ = rotate1(rgb_img_aug, elps)
                mask_after_r2 = rotate2(rotated_mask1, direction)
                rgb_after_r2  = rotate2(rgb_rot1,     direction)

    
                mask_fixed, rgb_fixed = ensure_u_upright(mask_after_r2, rgb_after_r2)

            
                mask_gray = cv2.cvtColor(mask_fixed, cv2.COLOR_BGR2GRAY)
                _, mask_bin = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                res = cv2.bitwise_and(rgb_fixed, rgb_fixed, mask=mask_bin)
                res = fit_box(res)

                out_path = os.path.join(des_dir, os.path.basename(mask_path))
                ok = cv2.imwrite(out_path, res)
                print("  -> saved:", out_path, "ok:", ok)

            except Exception as e:
                print("  ! Error on", mask_path, ":", repr(e))
                continue

# ---------------------------
# Walk us_split tree and mirror structure for accepted/fragmented
# ---------------------------

def process_us_split_tree(us_split_root, original_root, output_root,
                          categories=('accepted','fragmented'), mask_ext="*.png"):
    """
    us_split_root: root containing .../<rel_path>/(accepted|fragmented|rejected)/
    original_root: root containing original images in .../<rel_path>/
    output_root  : where to save patches, mirroring .../<rel_path>/(accepted|fragmented)/
    """
    us_split_root = os.path.abspath(us_split_root)
    original_root = os.path.abspath(original_root)
    output_root   = os.path.abspath(output_root)
    os.makedirs(output_root, exist_ok=True)

    print("[WALK] us_split_root:", us_split_root)
    hit = 0
    for dirpath, dirnames, _ in os.walk(us_split_root):
        leaf_name = os.path.basename(dirpath)
        if leaf_name not in categories:
            continue  

 
        rel_parent = os.path.relpath(os.path.dirname(dirpath), us_split_root)
      
        images_dir = os.path.join(original_root, rel_parent)
        if not os.path.isdir(images_dir):
            print("  ! Missing original images dir for", dirpath, "->", images_dir)
            continue

        
        dest_dir = os.path.join(output_root, rel_parent, leaf_name)
        os.makedirs(dest_dir, exist_ok=True)

        iterate_dir_save_accepted(dirpath, images_dir, dest_dir)
        hit += 1

    if hit == 0:
        print("[WARN] No category folders found under:", us_split_root)
    else:
        print(f"\n[DONE] Processed {hit} folder(s). Output root: {output_root}")

def main():
    parser = argparse.ArgumentParser(description="Extract oriented patches for accepted/fragmented masks, mirroring folder structure.")
    parser.add_argument("--us_split_root", required=True, help="Root with subfolders having accepted/ and fragmented/ (from the splitter).")
    parser.add_argument("--original_root", required=True, help="Root with original images mirroring the relative paths.")
    parser.add_argument("--output_root",  required=True, help="Destination root to save extracted patches with mirrored structure.")
    parser.add_argument("--categories", nargs="+", default=["accepted","fragmented"],
                        help="Which categories to process (default: accepted fragmented).")
    args = parser.parse_args()

    process_us_split_tree(args.us_split_root, args.original_root, args.output_root,
                          categories=tuple(args.categories))

if __name__ == "__main__":
    main()
