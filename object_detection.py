import torch
from PIL import Image
import pandas as pd

# Load the YOLOv5 large model for better accuracy
model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)

def detect_objects(image_path, conf_threshold=0.5):
    """
    Detect objects in an image using YOLOv5l with a confidence threshold.

    Parameters:
    - image_path (str): Path to the image file.
    - conf_threshold (float): Minimum confidence score to keep detections.

    Returns:
    - filtered_results (DataFrame): Filtered detections with bounding boxes and confidence scores.
    """
    # Load the image
    img = Image.open(image_path)

    # Perform object detection
    results = model(img)

    # Convert results to DataFrame and filter by confidence threshold
    filtered_results = results.pandas().xyxy[0]
    filtered_results = filtered_results[filtered_results['confidence'] >= conf_threshold]

    return filtered_results