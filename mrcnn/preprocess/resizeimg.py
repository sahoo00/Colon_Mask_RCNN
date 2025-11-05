from PIL import Image
import os

def resize_images_in_folder(source_folder, destination_folder, size=(2048,1536)):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    for filename in os.listdir('/data/space1/BooleanLab/Atishna/mask_cnn/Colon_Mask_RCNN/mouse'):
        print(filename)
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            source_path = os.path.join(source_folder, filename)
            destination_path = os.path.join(destination_folder, filename)
            with Image.open(source_path) as img:
                img = img.resize(size, Image.ANTIALIAS)
                img.save(destination_path)
                print(f"Resized {filename} and saved to {destination_path}")

source_folder = '/data/space1/BooleanLab/Atishna/mask_cnn/Colon_Mask_RCNN/mouse'
  # Replace with the path to your source folder
destination_folder = '/data/space1/BooleanLab/Atishna/mask_cnn/Colon_Mask_RCNN/mouse_resi'  # Replace with the path to your destination folder
resize_images_in_folder(source_folder, destination_folder)


