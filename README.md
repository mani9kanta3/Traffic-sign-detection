
# 🚦 Traffic Sign Detection for Autonomous Vehicles

This project focuses on detecting and recognizing traffic signs using deep learning models to support autonomous vehicle navigation. The goal is to accurately identify various traffic signs in real-time using YOLOv8 and TensorFlow pipelines.

## 📌 Key Features

- Trained using a custom dataset prepared with Roboflow
- Implemented YOLOv8 model for high-speed, real-time inference
- Data augmentation and annotation included
- Achieved robust detection across multiple traffic sign types
- Integrated images and pretrained weights (`yolov8n.pt`) for demonstration

## 🧰 Technologies Used

- Python
- YOLOv8 (Ultralytics)
- TensorFlow (for alternate model training)
- OpenCV
- Roboflow
- Jupyter Notebook

## 📁 Files and Folders

- `1.ipynb`, `2.ipynb`: Training and inference workflows
- `traffic sign detection.v1i.tensorflow.zip`: TensorFlow dataset format
- `traffic sign detection.v1i.yolov8.zip`: YOLOv8-ready dataset
- `yolov8n.pt`: Pretrained YOLOv8 nano model
- `images.png`, `multiple traffic signs.jpg`, etc.: Sample images for evaluation

## 🚀 How to Run

1. Install dependencies:
   ```bash
   pip install ultralytics opencv-python roboflow
   ```

2. Unzip the YOLOv8 dataset:
   ```bash
   unzip traffic\ sign\ detection.v1i.yolov8.zip
   ```

3. Run the training/inference notebooks (`1.ipynb`, `2.ipynb`) using Jupyter or VS Code.

## 📊 Results

Sample output demonstrates correct identification of multiple traffic signs with bounding boxes and class labels in real-time.

## 📂 Dataset

The dataset was exported from Roboflow in YOLOv8 and TensorFlow formats. It includes annotated images of traffic signs suitable for training and evaluation.
