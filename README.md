# YOLOv8 Glioblastoma (Brain Tumor) Segmentation Model

## Kerollos Lowandy

**Repository: glioblastoma-segmentation**

## GitHub Link
[https://github.com/Kerollosl/glioblastoma-segmentation](https://github.com/Kerollosl/glioblastoma-segmentation)

### Model Architecture
- YOLOv8

### Necessary Packages
- ultralytics: 8.2.76
- cv2: 4.8.0.76
- roboflow: 1.1.37
- yaml: 6.0.2
- IPython: 7.34.0

### Directions

1. In a Python notebook environment, upload the `/runs/segment/train5/weights/best.pt` file to allow the model to begin training with pre-trained weights. Run the `train_and_validate.ipynb` notebook. Note: This process is done in a notebook to leverage simplified downloading of the Roboflow dataset and use of a virtual GPU.
2. Once the notebook has been run, download the created `runs.zip` file. Unzip and upload to the same directory as the `test.py` script. Navigate in the runs directory to find the path for the new `best.pt` file. Change the path in the `test.py` script on line 20 to match this path. 
3. Run the `test.py` script. This runs the model on several test cases and segments predicted instances of glioblastoma. 
4. To add more test cases, upload an image to the `test_images/` subdirectory. The image should be an MRI brain scan image. The model works best on images from the top of ones head with a view that is perpendicular to the direction of one's eyes. The model confidence threshold has been set to a relatively low 0.2 as in a task such as this one, recall is more important than accuracy.

### Contents

- `test.py` - Program to load model and test it with test cases
- `train_and_validate.ipynb` - Program to download Roboflow dataset, load, train, and validate YOLOv8 model, and save weights and validation metrics to a zip file to be downloaded. 
- `/runs/segment` - Subdirectory of YOLOv8 model training and validation containing the training weights and validation metrics.
- `/runs/segment/train5/weights/best.pt` - Highest performing model weights file from previous trainings 
- `/test_images/` - Subdirectory containing images to test the model's performance.