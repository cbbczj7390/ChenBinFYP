import torch
from PIL import Image
import pandas as pd
from faster_rcnn_model import detect_objects_frcnn

# Load the YOLOv5 large model
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)

def detect_objects(image_path, model_type='yolov5', conf_threshold=0.5):
    """
    Detect objects in an image using YOLOv5 or Faster R-CNN.

    Parameters:
    - image_path (str): Path to the image file.
    - model_type (str): The model to use ('yolov5' or 'faster_rcnn').
    - conf_threshold (float): Minimum confidence score to keep detections.

    Returns:
    - results (DataFrame or list): Filtered detections with bounding boxes and confidence scores.
    """
    if model_type == 'yolov5':
        img = Image.open(image_path)
        results = yolo_model(img)
        filtered_results = results.pandas().xyxy[0]
        return filtered_results[filtered_results['confidence'] >= conf_threshold]
    elif model_type == 'faster_rcnn':
        return detect_objects_frcnn(image_path, conf_threshold)
    else:
        raise ValueError("Invalid model_type. Choose 'yolov5' or 'faster_rcnn'.")