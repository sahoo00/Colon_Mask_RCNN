# Expression Gradient of Cancer Suppressor Gene Found in Colon Crypt using Vision-AI
This repository details the framework implementation of the research findings from the paper titled "Expression Gradient of Cancer Suppressor Gene Found in Colon Crypt Using Vision-AI". The framework uses Keras, TensorFlow, and OpenCV. This robust pipeline is the first of its kind to perform U-shaped crypt segmentation. It accurately delineates crypts and glands within colon tissue images (histopathological) by creating bounding boxes and segmentation masks for each identified instance. The pipeline first detects and segments the crypts from the histopathological colon tissue image. Then it extracts the detected mask from the original image and applies advanced computer vision techniques to align the crypts top-to-bottom in a U-shape. At the end, compute the color intensities in the top and bottom portions of the extracted crypts. 


## Installation

1. Clone this repository
2. Install dependencies

   ```bash

   conda env create -f conda_environment.yml 
   conda activate base
   pip install histomicstk --find-links https://girder.github.io/large_image_wheels 

   ```

3. Run setup from the repository root directory
    ```bash

    python3 setup.py install

    ``` 

4. Download the dataset from this link: https://ucsdcloud-my.sharepoint.com/:f:/g/personal/dsahoo_ucsd_edu/EpwRsAH91HxFoj8FTnqh0NUB93MYYdji0hSWaDuMlBMaVQ?e=IKIZ2k and put it in the mrcnn/dataset folder

5. Download the inference dataset from this link: https://ucsdcloud-my.sharepoint.com/:f:/g/personal/dsahoo_ucsd_edu/EgOIJFepWBJFpFIY8TsocsgBT8NxJ8M7DfWHUlWV-Q1bcQ?e=Ljk1cr

6. Add this project to your pythonpath

   ```bash

   export PYTHONPATH="${PYTHONPATH}:/path/to/this/project/root"

   ```


# Training on Colon Crypts Dataset

Before training the model on the Colon Crypts and glands dataset, ensure that you have completed the installation process and successfully downloaded the dataset in the correct folder.

```bash
cd mrcnn/training/
python3 my_train_crypt.py -h
usage: my_train_crypt.py [-h] --dataset DATASET --dest DEST [--model MODEL]

arguments:
  -h, --help         show this help message and exit
  --dataset DATASET  path to the dataset, e.g.: dataset/Normalized_Images
  --dest DEST        name of the output model, e.g.:final.h5
  --model MODEL      path to the model, e.g.: logs/no_transfer/mask_rcnn_crypt_0060.h5
 
# Train a new model from the scratch
python3 my_train_crypt.py --dataset dataset/Normalized_Images --dest final.h5

# Train a new model starting from pretrained model
python3 my_train_crypt.py --dataset dataset/Normalized_Images --dest final.h5 --model logs/base_model.h5

```
# Prediction
For model prediction on the given dataset, you can use the following instructions:

```bash
cd mrcnn/inference/
python my_inference.py -h
usage: my_inference.py [-h] --dataset DATASET --model MODEL
 
arguments:
  -h, --help         show this help message and exit
  --dataset DATASET  path to the dataset, e.g.: dataset/Normalized_Images/
  --model MODEL      name of the model, e.g.:final.h5

# Prediction and generate the mask files
python3 my_inference.py --dataset=dataset/Normalized_Images/test/ --model=final.h5

```
# Running the Whole Pipeline
To reproduce the result from the paper, you need to normalize the color spectrum in the dataset using reinhard algorithm. 

```bash
cd mrcnn/preprocess/
python color_normalize_pipe.py -h
usage: color_normalize_pipe.py [-h] --ref REF --dest DEST --src SRC

arguments:
  -h, --help   show this help message and exit
  --ref REF    path to the reference image, e.g.: dataset/color_reference.png
  --dest DEST  Folder path of the output images, e.g.:dataset/Color_Normalized/
  --src SRC    Folder path of the source images, will process only .png images, e.g.: dataset/Raw_images/

# Normalize the dastaset images
python3 color_normalize_pipe.py 

```

Having the color normalized data, you need to apply hough transform to better align the images before the machine learning pipeline.

```bash
cd mrcnn/preprocess/
usage: hough_transform.py [-h] --dest DEST --src SRC

arguments:
  -h, --help   show this help message and exit
  --dest DEST  Root folder path of the dest images Exp: images/
  --src SRC    Root folder path of the source images (will only process png
               files) Exp: if you have images in images/data1/1.png
               images/data2/3.png you should pass: images/
# Transform the dataset
python3 hough_transform.py --src dataset/test_images/ --dest dataset/test_images/ 
```

After the image xy-axis align, now you can run the inference code and save the predictions in a different folder. It will create two folders, one having all predicted images and other predicted masks.

```bash
cd mrcnn/inference/

arguments:
  --dest DEST  Root folder path of the dest images e.g.: dataset/inf/
  --src SRC    Root folder path of the source images e.g.: dataset/test_images/

python3 my_inference.py --src=dataset/test_images/ --dest=dataset/inf/ --model=final.h5

```
Now there will be another folder with the predicted mask of the provided images, in the folder named "predicted_mask". From here you can call the next pipeline to rotate the U-shapes to align them from bottom to top for the futher analysis.

```bash
cd mrcnn/inference/
# It splits the predicted mask into three categories 1: Perfect U-shape 2: Fragmented U-shape 3: Wrong Prediction O-shape glands
arguments:
  --dest DEST  Root folder path of the dest images e.g.: dataset/us_split/
  --src SRC    Root directory containing inference results folders with a 'predicted_mask/' subfolder 

python U_shape_split.py --inf_root=dataset/inf/predicted_mask/ --out_root=us_split


```

```bash
cd mrcnn/inference/
python U-shape-bottom-pipeline.py -h
usage: U-shape-bottom-pipeline.py [-h] --src SRC

arguments:
  -h, --help  show this help message and exit
  --us_split_root SRC   Root with subfolders having "accepted/" and "fragmented/" from the splitter.
  --original_root SRC   Root with original images mirroring the relative paths.
  --output_root DEST    Destination root to save extracted patches with mirrored structure.
  --categories  DEST    Which categories to process "(default: accepted fragmented)".

# Bottom up alignment
python3 U-shape-bottom-pipeline.py 
```

Finally in the last step, the color spectrum needs to get measured from bottom to top. The following script will separate the image_mask, plot the blue vs brown color for each image and also aggregate all the images result together in the final result text file.

```bash
python color_detection.py -h
usage: color_detection.py [-h] --src SRC --dest DEST

optional arguments:
  -h, --help   show this help message and exit
  --src SRC    path to the dataset, Root containing subfolders with "accepted/" and/or "fragmented/". 
  --dest DEST  path to the final result text file, Exp: res.txt. The result
               table has the following columns in the tsv format: Folder_name,
               average of the blue intensity in the first 50% of the image,
               average blue in the last 50%, blue in the first 25%, blue in
               the last 25%, followed by the same column for average brown
               intensity. Also it saves the per image original image, nuclei_mask, blue and brown mask and color pattern. 

# Color detection values
python3 color_detection.py --src dataset/us_final/ --dest dataset/final/
```


## Citation
Use this bibtex to cite this paper and repo:
```
@misc{colon_maskrcnn_2025,
  title={Expression Gradient of Cancer Suppressor Gene Found in Colon Crypt Using Vision-AI},
  author={Mahdi Behroozikhah, Amitash Nanda, Soni Khandelwal, Atishna Samantaray, Dharanidhar Dang, Debashis Sahoo},
  year={2025},
}
```

