
# =========================
# UPDATED splitter (handles tiny holes; stricter O-rule)
# =========================

import os, shutil, glob
import numpy as np
import cv2
import argparse

from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label


MIN_OBJ_AREA       = 32          
CLOSE_KERNEL_SIZE  = 3          
HOLE_MIN_ABS       = 64         
HOLE_MIN_FRAC      = 0.01      
GOOD_U_ENDPTS_MIN  = 2           
GOOD_U_ENDPTS_MAX  = 4

def read_mask_binary(path):
    """Read mask → bool foreground, lightly cleaned for classification."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    _, binv = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binv = (binv > 0)


    if binv.any():
        binv = remove_small_objects(binv, min_size=MIN_OBJ_AREA)

    
    ksz = CLOSE_KERNEL_SIZE
    if ksz and ksz > 1:
        ker = np.ones((ksz, ksz), np.uint8)
        binv = cv2.morphologyEx(binv.astype(np.uint8)*255, cv2.MORPH_CLOSE, ker) > 0

    return binv

def count_endpoints(skel_bool):
    """Endpoints on 8-neighborhood skeleton: pixels with exactly one neighbor."""
    if skel_bool is None or skel_bool.sum() == 0:
        return 0
    skel_u8 = skel_bool.astype(np.uint8)
    kernel = np.array([[1,1,1],
                       [1,0,1],
                       [1,1,1]], dtype=np.uint8)
    nbrs = cv2.filter2D(skel_u8, ddepth=cv2.CV_16S, kernel=kernel, borderType=cv2.BORDER_CONSTANT)
    return int(np.sum((skel_u8 == 1) & (nbrs == 1)))

def count_large_holes(binary_mask):
    """
    Count holes but IGNORE tiny ones.
    Strategy: label the INVERTED mask; any component NOT touching the border
    is a hole. Only keep those with area >= max(HOLE_MIN_ABS, HOLE_MIN_FRAC * foreground_area).
    """
    if binary_mask is None:
        return 0, 0
    fg_area = int(binary_mask.sum())
    if fg_area == 0:
        return 0, 0

    min_hole_area = max(HOLE_MIN_ABS, int(HOLE_MIN_FRAC * fg_area))

    inv = ~binary_mask
    lbl = label(inv, connectivity=2)  
    nlabels = int(lbl.max())
    if nlabels == 0:
        return 0, 0

  
    border_labels = np.unique(np.concatenate([lbl[0,:], lbl[-1,:], lbl[:,0], lbl[:,-1]]))
    hole_labels = [i for i in range(1, nlabels+1) if i not in border_labels]

    total_holes = len(hole_labels)
    if total_holes == 0:
        return 0, 0

    large = 0
    for i in hole_labels:
        area = int((lbl == i).sum())
        if area >= min_hole_area:
            large += 1
    return large, total_holes

def classify_mask(binary_mask):
    """
    Returns one of: 'good_u', 'fragmented_u', 'o_shape', 'rejected'
    O-shape (loop): single component AND has ≥1 large hole AND endpoints ≤ 1
    Fragmented U:   no large holes AND (components ≥2 OR endpoints ≥3)
    Good U:         no large holes AND single component AND endpoints in [2,4]
    """
    if binary_mask is None or binary_mask.sum() == 0:
        return "rejected", {"reason": "empty"}

    num_comp = int(label(binary_mask, connectivity=2).max())
    large_holes, total_holes = count_large_holes(binary_mask)

    skel = skeletonize(binary_mask)
    endpoints = count_endpoints(skel)

    if (num_comp == 1) and (large_holes >= 1) and (endpoints <= 1):
        return "o_shape", {"components": num_comp, "large_holes": large_holes,
                           "total_holes": total_holes, "endpoints": endpoints}

   
    if (large_holes == 0) and ((num_comp >= 2) or (endpoints >= 3)):
        return "fragmented_u", {"components": num_comp, "large_holes": large_holes,
                                "total_holes": total_holes, "endpoints": endpoints}


    if (large_holes == 0) and (num_comp >= 1) and (GOOD_U_ENDPTS_MIN <= endpoints <= GOOD_U_ENDPTS_MAX):
        return "good_u", {"components": num_comp, "large_holes": large_holes,
                          "total_holes": total_holes, "endpoints": endpoints}


    return "rejected", {"components": num_comp, "large_holes": large_holes,
                        "total_holes": total_holes, "endpoints": endpoints, "reason": "uncertain"}

def split_directory_single(mask_dir, out_dir, pattern="*.png"):
    """
    Scan one masks folder and split into accepted/fragmented/rejected.
    Each original mask is COPIED (not moved). Safe to re-run.
    """
    acc_dir = os.path.join(out_dir, "accepted")
    frag_dir = os.path.join(out_dir, "fragmented")
    rej_dir  = os.path.join(out_dir, "rejected")
    os.makedirs(acc_dir, exist_ok=True)
    os.makedirs(frag_dir, exist_ok=True)
    os.makedirs(rej_dir,  exist_ok=True)

    paths = sorted(glob.glob(os.path.join(mask_dir, pattern)))
    print(f"[INFO] Found {len(paths)} mask(s) in: {mask_dir}")

    n_good = n_frag = n_rej = 0
    for p in paths:
        binv = read_mask_binary(p)
        cls, meta = classify_mask(binv)
        base = os.path.basename(p)

        if cls == "good_u":
            dst = os.path.join(acc_dir, base); n_good += 1
        elif cls == "fragmented_u":
            dst = os.path.join(frag_dir, base); n_frag += 1
        else:  
            dst = os.path.join(rej_dir, base);  n_rej  += 1

        shutil.copy2(p, dst)
        print(f"  • {base:>30s} -> {cls:12s}  {meta}")

    print("\n[SUMMARY]")
    print(f"  accepted (good U):   {n_good}")
    print(f"  fragmented U:        {n_frag}")
    print(f"  rejected (O/other):  {n_rej}")
    print(f"  Output root: {out_dir}")

# ---------------------------------------------------------------------
# Mirror the directory structure under 'inf' and split each leaf
# that contains a 'predicted_mask/' folder.
# ---------------------------------------------------------------------

def split_tree(inf_root, out_root, mask_subdir="predicted_mask", pattern="*.png"):
    """
    Walks `inf_root` recursively. For every directory that contains a
    subfolder named `mask_subdir`, runs `split_directory_single` on that
    mask folder and writes results to:
        out_root / <relative path from inf_root to the parent of mask_subdir>
    Inside that mirrored leaf it creates: accepted/, fragmented/, rejected/.
    """
    inf_root = os.path.abspath(inf_root)
    out_root = os.path.abspath(out_root)
    os.makedirs(out_root, exist_ok=True)

    print(f"[WALK] Scanning: {inf_root}")
    hits = 0
    for dirpath, dirnames, _ in os.walk(inf_root):
        if mask_subdir in dirnames:
            mask_dir = os.path.join(dirpath, mask_subdir)
   
            if not glob.glob(os.path.join(mask_dir, pattern)):
                continue
            rel_dir = os.path.relpath(dirpath, inf_root)  
            out_dir = os.path.join(out_root, rel_dir)
            os.makedirs(out_dir, exist_ok=True)
            print(f"\n[PROCESS] {mask_dir}")
            print(f"[OUTPUT ] {out_dir}")
            split_directory_single(mask_dir, out_dir, pattern=pattern)
            hits += 1

    if hits == 0:
        print("[WARN] No '{}' folders with masks found under: {}".format(mask_subdir, inf_root))
    else:
        print(f"\n[DONE] Processed {hits} mask folder(s). Output root: {out_root}")

def main():
    parser = argparse.ArgumentParser(description="Split predicted masks into accepted/fragmented/rejected, mirroring folder structure.")
    parser.add_argument("--inf_root",  required=True, help="Root directory containing inference results (folders with a 'predicted_mask/' subfolder).")
    parser.add_argument("--out_root",  required=True, help="Output root where the mirrored split folders will be created.")
    args = parser.parse_args()

    split_tree(args.inf_root, args.out_root, mask_subdir="predicted_mask", pattern="*.png")

if __name__ == "__main__":
    main()
