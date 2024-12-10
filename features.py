from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import numpy as np

# Load ResNet50 model with batch normalization layers
base_model = ResNet50(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

def extract_features(img_path):
    """
    Extracts and returns the feature vector of an image using ResNet50.

    Parameters:
    - img_path (str): The file path of the image.

    Returns:
    - numpy.ndarray: The normalized feature vector.
    """
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = model.predict(img_data)
    return features.flatten() / np.linalg.norm(features.flatten())  # Normalize features