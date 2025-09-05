# Object Detection with YOLOv4

This project implements **real-time object detection** using the YOLOv4 model.  
It loads pretrained YOLOv4 weights and configuration to detect objects from images or video streams.

## 🚀 Features
- YOLOv4 architecture
- Pretrained on COCO dataset
- Detects 80+ object categories
- Python + OpenCV implementation

## 📂 Project Structure
detection_objet/
│── object_detection.py # Main detection script
│── yolov4.cfg # Model configuration
│── yolov4.weights # Pretrained weights (not uploaded to GitHub)
│── coco.names # COCO class labels


## ⚡ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/detection_objet.git
   cd detection_objet


Install dependencies:

pip install opencv-python numpy


Download YOLOv4 pretrained weights inside the same folder (if not included):

wget https://pjreddie.com/media/files/yolov4.weights


▶️ Usage

Just save a picture inside the folder and named input.png and run the code with any compiler