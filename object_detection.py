import torch
from PIL import Image
from faster_rcnn_model import detect_objects_frcnn

# Load the YOLOv5 large model
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)

def detect_objects(image_path, model_type='faster_rcnn', conf_threshold=0.5):
    """
    Detect objects in an image using YOLOv5 or Faster R-CNN.

    For `update_tags` functionality, we force the use of Faster R-CNN.

    Parameters:
    - image_path (str): Path to the image file.
    - model_type (str): The model to use ('yolov5' or 'faster_rcnn').
    - conf_threshold (float): Minimum confidence score to keep detections.

    Returns:
    - detections (list of dict): Detected objects with names and confidence scores.
    """
    if model_type == 'faster_rcnn':
        # Force Faster R-CNN detection for update_tags
        try:
            detections = detect_objects_frcnn(image_path, conf_threshold)

            # Process Faster R-CNN results into a consistent format
            processed_detections = []
            if isinstance(detections, list):
                for obj in detections:
                    if isinstance(obj, dict) and 'name' in obj and 'confidence' in obj:
                        processed_detections.append({
                            'name': obj['name'],
                            'confidence': obj['confidence']
                        })
            return processed_detections

        except Exception as e:
            print(f"Error during Faster R-CNN detection: {e}")
            return []

    elif model_type == 'yolov5':
        # YOLOv5 detection
        try:
            img = Image.open(image_path)
            results = yolo_model(img)
            filtered_results = results.pandas().xyxy[0]
            detections = filtered_results[filtered_results['confidence'] >= conf_threshold]

            # Process YOLOv5 results into a consistent format
            processed_detections = [
                {'name': row['name'], 'confidence': row['confidence']}
                for _, row in detections.iterrows()
            ]
            return processed_detections

        except Exception as e:
            print(f"Error during YOLOv5 detection: {e}")
            return []

    else:
        raise ValueError("Invalid model_type. Choose 'yolov5' or 'faster_rcnn'.")