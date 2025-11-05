import os
import glob
import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d  


def show_image_plt(im1, im2, im3, im4, blue_count, brown_count, labels, save_path):
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5, figsize=(12, 3))
    ax1.imshow(im1); ax1.set_title(labels[0]); ax1.axis('off')
    ax2.imshow(im2); ax2.set_title(labels[1]); ax2.axis('off')
    ax3.imshow(im3, cmap='gray'); ax3.set_title(labels[2]); ax3.axis('off')
    ax4.imshow(im4, cmap='gray'); ax4.set_title(labels[3]); ax4.axis('off')
    ax5.plot(blue_count[::-1],  range(len(blue_count[::-1])),  label='blue')
    ax5.plot(brown_count[::-1], range(len(brown_count[::-1])), label='brown')
    ax5.set_title(labels[4]); ax5.legend(); ax5.axis('off')
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def mask_builder(image_bgr, hl, hh, sl, sh, vl, vh):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([hl, sl, vl], dtype=np.uint8)
    upper_bound = np.array([hh, sh, vh], dtype=np.uint8)
    return cv2.inRange(hsv, lower_bound, upper_bound)


def process_accepted_fragmented(parent_dir, dest_dir, target_size=(400, 1200)):
    """
    parent_dir: folder that contains 'accepted/' and/or 'fragmented/' (or 'fragments/')
    dest_dir  : mirror of parent_dir under output root
    Saves per-image artifacts + all_in_one.pdf + final.txt in dest_dir.
    """
 
    cat_dirs = []
    for name in ('accepted', 'fragmented', 'fragments'):
        p = os.path.join(parent_dir, name)
        if os.path.isdir(p):
            cat_dirs.append(p)
    if not cat_dirs:
        return  

    os.makedirs(dest_dir, exist_ok=True)
    mask_and_dir = os.path.join(dest_dir, "mask_and")
    bbm_dir      = os.path.join(dest_dir, "blue_brown_mask")
    os.makedirs(mask_and_dir, exist_ok=True)
    os.makedirs(bbm_dir, exist_ok=True)

    
    W, H = target_size               # width, height
    all_blue  = np.zeros(H, dtype=np.float64)
    all_brown = np.zeros(H, dtype=np.float64)
    total_images = 0

    for cat in cat_dirs:
        im_names = sorted(glob.glob(os.path.join(cat, "*.png")))
        for im_name in im_names:
            img = cv2.imread(im_name, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue

          
            img = cv2.resize(img, (W, H))
            img_original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if np.all(gray == 0):
                continue

        
            nz = gray[gray != 0]
            if nz.size == 0:
                continue
            thr, _ = cv2.threshold(nz, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            nuclei_mask = np.logical_and(0 < gray, gray <= thr).astype(np.uint8)

     
            kernel = np.ones((2, 2), np.uint8)
            opening = cv2.morphologyEx(nuclei_mask, cv2.MORPH_OPEN, kernel)
            closing = cv2.morphologyEx(opening,     cv2.MORPH_CLOSE, kernel)

           
            img_nuclei_only = cv2.bitwise_and(img, img, mask=closing)
            cv2.imwrite(os.path.join(mask_and_dir, os.path.basename(im_name)), img_nuclei_only)
            img_rgb_nuclei_only = cv2.cvtColor(img_nuclei_only, cv2.COLOR_BGR2RGB)

          
            brown_mask = np.logical_or(
                mask_builder(img_nuclei_only, 0,   40, 1, 254, 1, 254) > 0,
                mask_builder(img_nuclei_only, 150, 180, 1, 254, 1, 254) > 0
            ).astype(np.uint8) * 255
            blue_mask  = mask_builder(img_nuclei_only, 80, 140, 1, 254, 1, 254)

        
            brown_count = np.sum(brown_mask == 255, axis=1).astype(np.float64)
            blue_count  = np.sum(blue_mask  == 255, axis=1).astype(np.float64)

         
            brown_s = gaussian_filter1d(brown_count, sigma=15)
            blue_s  = gaussian_filter1d(blue_count,  sigma=15)

       
            total = blue_s + brown_s + 1e-6
            blue_norm  = blue_s  / total
            brown_norm = brown_s / total

          
            plot_path = os.path.join(bbm_dir, os.path.splitext(os.path.basename(im_name))[0] + '.pdf')
            show_image_plt(
                img_original, img_rgb_nuclei_only, blue_mask, brown_mask,
                blue_norm, brown_norm,
                labels=["original", "nuclei_mask", "blue", "brown", "color pattern"],
                save_path=plot_path
            )

         
            all_blue  += blue_norm
            all_brown += brown_norm
            total_images += 1

  
    if total_images > 0:
        all_brown_smooth = gaussian_filter1d(all_brown, sigma=10)
        all_blue_smooth  = gaussian_filter1d(all_blue,  sigma=10)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(all_blue_smooth[::-1],  range(len(all_blue_smooth[::-1])),  label='blue')
        ax.plot(all_brown_smooth[::-1], range(len(all_brown_smooth[::-1])), label='brown')
        ax.legend()
        ax.set_xlabel('Summed Color Intensity (normalized per image, then summed)')
        ax.set_ylabel('Image Height (Top to Bottom)')
        ax.set_title(f'Aggregated Color Profile\n{os.path.basename(parent_dir)}')
        ax.grid(True)
        fig.savefig(os.path.join(dest_dir, 'all_in_one.pdf'), bbox_inches='tight')
        plt.close(fig)


    base_name  = os.path.basename(parent_dir.rstrip(os.sep))
    final_path = os.path.join(dest_dir, 'final.txt')
    with open(final_path, 'w') as f:
        headers = [
            'directory_name', 'number_of_images',
            'avg_blue_top_50%',  'avg_blue_bottom_50%',
            'avg_blue_top_25%',  'avg_blue_bottom_25%',
            'avg_brown_top_50%', 'avg_brown_bottom_50%',
            'avg_brown_top_25%', 'avg_brown_bottom_25%',
        ]
        f.write('\t'.join(headers) + '\n')

        def halves_quarters(arr):
            n = len(arr)
            if n == 0:
                return (0.0, 0.0, 0.0, 0.0)
            return (
                float(np.mean(arr[: n//2])),
                float(np.mean(arr[n//2 :])),
                float(np.mean(arr[: n//4])),
                float(np.mean(arr[(3*n)//4 :]))
            )

        b_top50,  b_bot50,  b_top25,  b_bot25  = halves_quarters(all_blue)
        br_top50, br_bot50, br_top25, br_bot25 = halves_quarters(all_brown)

        row = [
            base_name, str(total_images),
            str(b_top50),  str(b_bot50),  str(b_top25),  str(b_bot25),
            str(br_top50), str(br_bot50), str(br_top25), str(br_bot25),
        ]
        f.write('\t'.join(row) + '\n')

    print(f"[DONE] {parent_dir} -> {dest_dir}  (images: {total_images})")


def walk_and_process(src_root, dest_root):
    """
    For every directory under src_root that contains 'accepted/' and/or 'fragmented/' (or 'fragments/'),
    collect all PNGs in those folders and save artifacts + final.txt under dest_root,
    mirroring the relative path of that parent directory.
    """
    src_root  = os.path.abspath(src_root)
    dest_root = os.path.abspath(dest_root)
    os.makedirs(dest_root, exist_ok=True)

    hits = 0
    for dirpath, dirnames, filenames in os.walk(src_root):
        has_acc = 'accepted'   in dirnames
        has_fra = 'fragmented' in dirnames or 'fragments' in dirnames
        if not (has_acc or has_fra):
            continue

        rel = os.path.relpath(dirpath, src_root)
        out_dir = os.path.join(dest_root, rel)
        os.makedirs(out_dir, exist_ok=True)

        process_accepted_fragmented(dirpath, out_dir)
        hits += 1

    if hits == 0:
        print(f"[WARN] No folders with 'accepted/' or 'fragmented/' under: {src_root}")
    else:
        print(f"[SUMMARY] Wrote artifacts for {hits} folder(s) under: {dest_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("color_detection.py")
    parser.add_argument("--src",  required=True, help="Root containing subfolders with accepted/ and/or fragmented/")
    parser.add_argument("--dest", required=True, help="Output root to mirror structure and write artifacts + final.txt")
    args = parser.parse_args()

    walk_and_process(args.src, args.dest)

