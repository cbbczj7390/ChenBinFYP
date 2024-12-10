import os
from features import extract_features
from database import init_db, save_image_features

# Initialize the database
init_db()

# Directory containing the initial images
IMAGE_FOLDER = 'initial_images'

# Function to process images in a folder and its subfolders
def process_images(folder):
    for root, _, files in os.walk(folder):
        for filename in files:
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, filename)
                features = extract_features(img_path)
                save_image_features(img_path, features)
                print(f"Added {filename} to the database.")

# Process images in the initial_images folder
process_images(IMAGE_FOLDER)