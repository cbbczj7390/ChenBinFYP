import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image

# Load Faster R-CNN model (pre-trained on COCO dataset)
faster_rcnn_model = fasterrcnn_resnet50_fpn(pretrained=True)
faster_rcnn_model.eval()

# COCO dataset class labels
COCO_CLASSES = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat',
    10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
    18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
    28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball',
    38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
    44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
    54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake',
    62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'TV', 73: 'laptop',
    74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
    82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier',
    90: 'toothbrush'
}

def detect_objects_frcnn(image_path, conf_threshold=0.5):
    """
    Detect objects in an image using Faster R-CNN with a confidence threshold.

    Parameters:
    - image_path (str): Path to the image file.
    - conf_threshold (float): Minimum confidence score to keep detections.

    Returns:
    - list: Filtered detections with bounding boxes, labels, and confidence scores.
    """
    img = Image.open(image_path).convert('RGB')
    img_tensor = F.to_tensor(img).unsqueeze(0)

    with torch.no_grad():
        predictions = faster_rcnn_model(img_tensor)

    detections = []
    for i, score in enumerate(predictions[0]['scores']):
        if score >= conf_threshold:
            box = predictions[0]['boxes'][i].tolist()
            class_id = predictions[0]['labels'][i].item()
            label = COCO_CLASSES.get(class_id, 'Unknown')
            detections.append({
                'box': box,
                'name': label,
                'confidence': score.item()
            })

    return detections