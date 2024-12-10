import os
from features import extract_features
from database import save_image_features, init_db

def add_images_from_directory(directory):
    """
    Adds images from the specified directory and its subdirectories to the database.
    
    Parameters:
    - directory (str): Path to the directory containing images.
    """
    # Initialize the database
    init_db()

    # Loop through all files in the directory and subdirectories
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                try:
                    print(f"Processing: {img_path}")
                    features = extract_features(img_path)
                    save_image_features(img_path, features)
                    print(f"Added: {img_path}")
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

if __name__ == '__main__':
    # Path to the initial_images folder
    directory = os.path.join('static', 'initial_images')
    add_images_from_directory(directory)