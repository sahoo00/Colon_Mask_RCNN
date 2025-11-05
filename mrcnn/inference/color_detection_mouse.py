
import os, glob, argparse
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ttest_ind


def mask_builder(image_bgr, hl, hh, sl, sh, vl, vh):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([hl, sl, vl], dtype=np.uint8)
    upper = np.array([hh, sh, vh], dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)

def nuclei_mask_bgr(img_bgr):
    """Otsu on nonzero gray + open/close (same spirit as your pipeline)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if np.all(gray == 0): 
        return np.zeros_like(gray, np.uint8)
    nz = gray[gray != 0]
    if nz.size == 0:
        return np.zeros_like(gray, np.uint8)
    thr, _ = cv2.threshold(nz, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    nm = np.logical_and(0 < gray, gray <= thr).astype(np.uint8)
    k = np.ones((2,2), np.uint8)
    nm = cv2.morphologyEx(nm, cv2.MORPH_OPEN,  k)
    nm = cv2.morphologyEx(nm, cv2.MORPH_CLOSE, k)
    return nm

def brown_intensity_topminusbottom(img_bgr, target_size=(400,1200), smooth_sigma=15):
    """
    Compute Brown Intensity (top25% − bottom25%) for a single image.
    Uses your HSV ranges and row-wise smoothing/normalization.
    """
    H, W = target_size  
    img = cv2.resize(img_bgr, (W, H), interpolation=cv2.INTER_AREA)

    nm = nuclei_mask_bgr(img)
    img_nuclei = cv2.bitwise_and(img, img, mask=nm)


    brown_mask = np.logical_or(
        mask_builder(img_nuclei, 0,   40, 1, 254, 1, 254) > 0,
        mask_builder(img_nuclei, 150, 180, 1, 254, 1, 254) > 0
    ).astype(np.uint8) * 255
    blue_mask  = mask_builder(img_nuclei, 80, 140, 1, 254, 1, 254)

    brown_count = np.sum(brown_mask == 255, axis=1).astype(np.float64)
    blue_count  = np.sum(blue_mask  == 255, axis=1).astype(np.float64)

    brown_s = gaussian_filter1d(brown_count, sigma=smooth_sigma)
    blue_s  = gaussian_filter1d(blue_count,  sigma=smooth_sigma)

    total = brown_s + blue_s
    total[total == 0] = 1e-6
    brown_norm = brown_s / total

    n = len(brown_norm)
    if n == 0:
        return 0.0
    top25    = float(np.mean(brown_norm[: n//4]))
    bottom25 = float(np.mean(brown_norm[(3*n)//4 :]))
    return top25 - bottom25


def find_patch_dir(root, expect='b'):
    """
    root: path to protocol root. Inside, we expect a single patches dir:
      - for boiling:  a folder containing 'patch_b_us' (case-insensitive)
      - for pressure: a folder containing 'patch_p_us' (case-insensitive)
    If the user passes the patch dir directly, we accept it as-is.
    """
    root = os.path.abspath(root)
    last = os.path.basename(root).lower()
    if (expect == 'b' and 'patch_b_us' in last) or (expect == 'p' and 'patch_p_us' in last):
        return root  

 
    candidates = []
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if not os.path.isdir(p):
            continue
        low = name.lower()
        if expect == 'b' and 'patch_b_us' in low:
            candidates.append(p)
        if expect == 'p' and 'patch_p_us' in low:
            candidates.append(p)
    if len(candidates) == 0:
        raise FileNotFoundError(f"No expected patches folder found under {root} (expect={'patch_b_us' if expect=='b' else 'patch_p_us'})")
    if len(candidates) > 1:

        candidates.sort()
    return candidates[0]

def collect_metrics_from_patch_dir(patch_dir):
    """
    From a patch dir that contains accepted/ and fragmented/ subfolders,
    gather all PNGs and compute per-image Brown Intensity.
    """
    vals = []
   
    for leaf in ('accepted', 'fragmented'):
        leaf_dir = os.path.join(patch_dir, leaf)
        if not os.path.isdir(leaf_dir):
            continue
        for p in glob.glob(os.path.join(leaf_dir, "*.png")):
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                continue
            try:
                val = brown_intensity_topminusbottom(img)
                vals.append((p, val))
            except Exception:
             
                continue
    return vals  


def main(boiling_root, pressure_root, out_dir, plot=False):
    os.makedirs(out_dir, exist_ok=True)


    b_patch = find_patch_dir(boiling_root, expect='b')
    p_patch = find_patch_dir(pressure_root, expect='p')

    print("[INFO] Boiling patch dir :", b_patch)
    print("[INFO] Pressure patch dir:", p_patch)

    b_list = collect_metrics_from_patch_dir(b_patch)
    p_list = collect_metrics_from_patch_dir(p_patch)
    b_vals = np.array([v for _, v in b_list], dtype=float)
    p_vals = np.array([v for _, v in p_list], dtype=float)

    print(f"[INFO] Counts -> Boiling: {len(b_vals)}   Pressure: {len(p_vals)}")
    print(f"[INFO] Means  -> Boiling: {b_vals.mean():.6f}   Pressure: {p_vals.mean():.6f}")


    df = pd.DataFrame({
        "protocol": (["boiling"]*len(b_vals)) + (["pressure"]*len(p_vals)),
        "image_path": [p for p,_ in b_list] + [p for p,_ in p_list],
        "Brown Intensity(25% top - 25% bottom)": np.r_[b_vals, p_vals]
    })
    per_image_path = os.path.join(out_dir, "per_image_brown_intensity.tsv")
    df.to_csv(per_image_path, sep="\t", index=False)
    print(f"[SAVE] {per_image_path}")

    # stats
    if len(b_vals) >= 2 and len(p_vals) >= 2:
        t, p = ttest_ind(b_vals, p_vals, equal_var=False)
    else:
        t, p = np.nan, np.nan
        print("[WARN] Need ≥2 per group for a valid Welch t-test; reporting NaN.")

    summary = pd.DataFrame([{
        "n_boiling": len(b_vals),
        "n_pressure": len(p_vals),
        "mean_boiling": float(np.mean(b_vals) if len(b_vals) else np.nan),
        "mean_pressure": float(np.mean(p_vals) if len(p_vals) else np.nan),
        "diff_mean_boiling_minus_pressure": float((np.mean(b_vals) - np.mean(p_vals)) if len(b_vals) and len(p_vals) else np.nan),
        "welch_t": float(t) if np.isfinite(t) else np.nan,
        "p_value_welch": float(p) if np.isfinite(p) else np.nan
    }])
    summary_path = os.path.join(out_dir, "summary_and_pvalue.tsv")
    summary.to_csv(summary_path, sep="\t", index=False)
    print(f"[SAVE] {summary_path}")

    # optional plot
    if plot and len(df) > 0:
        try:
            import seaborn as sns
            import matplotlib.pyplot as plt
            sns.set(style="whitegrid", context="notebook")
            plot_df = df.copy()
            plot_df["protocol"] = plot_df["protocol"].map({"boiling":"Boiling","pressure":"Pressure"})
            fig, ax = plt.subplots(figsize=(6.5,5), dpi=300)
            order = ["Boiling","Pressure"]
            sns.violinplot(data=plot_df, x="protocol", y="Brown Intensity(25% top - 25% bottom)",
                           order=order, inner="quartile", linewidth=0.8, cut=0, ax=ax)
            sns.swarmplot(data=plot_df, x="protocol", y="Brown Intensity(25% top - 25% bottom)",
                          order=order, color="k", alpha=0.35, size=4, ax=ax)
            ax.set_xlabel("Protocol")
            ax.set_ylabel("Brown Intensity (top 25% − bottom 25%)")
            title = "Boiling vs Pressure"
            if np.isfinite(p):
                title += f"  (Welch p = {p:.3g})"
            ax.set_title(title)
            plt.tight_layout()
            out_png = os.path.join(out_dir, "violin_boiling_vs_pressure.png")
            plt.savefig(out_png, dpi=300)
            plt.close(fig)
            print(f"[SAVE] {out_png}")
        except Exception as e:
            print("[WARN] Plotting skipped:", e)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compute per-image Brown Intensity (top25%−bottom25%) and Welch p-value for boiling vs pressure.")
    ap.add_argument("--boiling_root",  required=True, help="Path to boiling root (contains patch_b_us) OR directly to patch_b_us")
    ap.add_argument("--pressure_root", required=True, help="Path to pressure root (contains patch_p_us) OR directly to patch_p_us")
    ap.add_argument("--out",           required=True, help="Output directory for TSVs (and optional plot)")
    ap.add_argument("--plot",          action="store_true", help="Save violin+swarm plot")
    args = ap.parse_args()

    main(args.boiling_root, args.pressure_root, args.out, plot=args.plot)
