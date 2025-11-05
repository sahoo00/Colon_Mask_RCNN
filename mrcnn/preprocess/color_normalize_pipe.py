
import histomicstk as htk
import cv2
import os
import argparse


def reinhard_normalization_recursive(src_root, dest_root, ref_img_path):
    """
    Applies Reinhard color normalization to all images within a source directory
    and its subdirectories, saving them to a destination directory with a mirrored
    structure.
    """
 
    print("Reading reference image...")
    ref_img = cv2.imread(ref_img_path)
    if ref_img is None:
        print(f"Error: Could not read reference image at {ref_img_path}")
        return
    mean_ref, std_ref = htk.preprocessing.color_conversion.lab_mean_std(ref_img)
    print("Reference statistics calculated.")

    valid_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')


    for dirpath, dirnames, filenames in os.walk(src_root):
        image_files = [f for f in filenames if f.lower().endswith(valid_extensions)]
        
        if not image_files:
            continue

        relative_path = os.path.relpath(dirpath, src_root)
     
        dest_dir = os.path.join(dest_root, relative_path)
      
        os.makedirs(dest_dir, exist_ok=True)
        print(f"\nProcessing directory: {dirpath}")
        
      
        for filename in image_files:
            input_path = os.path.join(dirpath, filename)
            

            base_filename = os.path.splitext(filename)[0]
            output_filename = f"{base_filename}.png"
            output_path = os.path.join(dest_dir, output_filename)
            
            print(f"  -> Normalizing {filename}...")

         
            img_input = cv2.imread(input_path)
            if img_input is None:
                print(f"    - Warning: Could not read {input_path}. Skipping.")
                continue

           
            im_nmzd = htk.preprocessing.color_normalization.reinhard(img_input, mean_ref, std_ref)
            
  
            cv2.imwrite(output_path, im_nmzd)

    print("\nNormalization complete!")


if __name__ == '__main__':

    parser = argparse.ArgumentParser("color_normalize_pipe.py")
    parser.add_argument("--ref", help="Path to the reference image. Exp: dataset/color_reference.png", type=str, required=True)
    parser.add_argument("--dest", help="Root folder for the output images. Exp: dataset/Color_Normalized/", type=str, required=True)
    parser.add_argument("--src", help="Root folder of the source images. Exp: dataset/Raw_images/", type=str, required=True)
    args = parser.parse_args()

    reinhard_normalization_recursive(args.src, args.dest, args.ref)