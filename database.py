import sqlite3
import numpy as np
import os

def init_db():
    """
    Initializes the database by creating a table for storing images, features, and tags.
    """
    conn = sqlite3.connect('images.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS images (
                 id INTEGER PRIMARY KEY,
                 path TEXT UNIQUE,
                 features BLOB,
                 tags TEXT)''')
    conn.commit()
    conn.close()
    print("Database initialized successfully.")

def get_all_images():
    """
    Retrieves all image paths from the database.
    """
    conn = sqlite3.connect('images.db')
    c = conn.cursor()
    c.execute("SELECT path FROM images")
    rows = c.fetchall()
    conn.close()
    return [row[0] for row in rows]

def save_image_features(img_path, features, tags):
    """
    Saves the image path, feature vector, and tags to the database.

    Parameters:
    - img_path (str): The file path of the image.
    - features (numpy.ndarray): The feature vector of the image.
    - tags (list): A list of tags/keywords associated with the image.
    """
    conn = sqlite3.connect('images.db')
    c = conn.cursor()
    try:
        c.execute("INSERT OR IGNORE INTO images (path, features, tags) VALUES (?, ?, ?)", 
                  (img_path, features.tobytes(), ','.join(tags)))
        conn.commit()
        print(f"Image features saved successfully for: {img_path}")
    except sqlite3.Error as e:
        print(f"Error saving image features for {img_path}: {e}")
    finally:
        conn.close()

def query_database(features, uploaded_path, threshold=0.3, debug=True):
    """
    Queries the database to find images similar to the uploaded image.

    Parameters:
    - features (numpy.ndarray): The feature vector of the uploaded image.
    - uploaded_path (str): The file path of the uploaded image.
    - threshold (float): The similarity threshold.
    - debug (bool): Whether to print debug information.

    Returns:
    - List of tuples containing image paths and similarity scores.
    """
    conn = sqlite3.connect('images.db')
    c = conn.cursor()
    c.execute("SELECT path, features FROM images")
    rows = c.fetchall()
    conn.close()

    distances = []
    for row in rows:
        db_path = row[0]
        db_features = np.frombuffer(row[1], dtype=np.float32)

        # Normalize the feature vectors
        db_features = db_features / np.linalg.norm(db_features)
        normalized_features = features / np.linalg.norm(features)

        # Calculate cosine similarity
        similarity = np.dot(normalized_features, db_features)
        if similarity >= threshold:
            distances.append((db_path, similarity))

        if debug:
            print(f"Comparing with: {db_path}, Similarity: {similarity:.4f}")

    return distances

def search_by_keyword(keyword):
    """
    Searches the database for images with tags containing the specified keyword.

    Parameters:
    - keyword (str): The keyword to search for in the tags.

    Returns:
    - List of dictionaries containing image paths and associated tags.
    """
    conn = sqlite3.connect('images.db')
    c = conn.cursor()
    try:
        c.execute("SELECT path, tags FROM images WHERE tags LIKE ?", (f"%{keyword}%",))
        results = c.fetchall()
        return [{'path': row[0], 'tags': row[1].split(',')} for row in results]
    except sqlite3.Error as e:
        print(f"Error searching by keyword '{keyword}': {e}")
        return []
    finally:
        conn.close()

def update_tags_for_existing_images(detect_objects_func, model_type='yolov5', conf_threshold=0.5, limit=None):
    """
    Updates all existing images in the database with tags generated by object detection.

    Parameters:
    - detect_objects_func (function): The object detection function to use.
    - model_type (str): The model type to use for object detection.
    - conf_threshold (float): The confidence threshold for detection.
    - limit (int): Maximum number of images to process (default is None, meaning no limit).
    """
    _update_tags(detect_objects_func, model_type, conf_threshold, condition="1=1", 
                 message="All images updated with tags.", limit=limit)

def update_tags_for_images_with_empty_tags(detect_objects_func, model_type='yolov5', conf_threshold=0.5, limit=None):
    """
    Updates tags only for images with empty tags in the database by running object detection.

    Parameters:
    - detect_objects_func (function): The object detection function to use.
    - model_type (str): The model type to use for object detection.
    - conf_threshold (float): The confidence threshold for detection.
    - limit (int): Maximum number of images to process (default is None, meaning no limit).
    """
    _update_tags(detect_objects_func, model_type, conf_threshold, 
                 condition="tags IS NULL OR tags = ''", 
                 message="Images with empty tags updated successfully.", limit=limit)

def _update_tags(detect_objects_func, model_type, conf_threshold, condition, message, limit=None):
    """
    Helper function to update tags based on a condition.

    Parameters:
    - detect_objects_func (function): The object detection function to use.
    - model_type (str): The model type to use.
    - conf_threshold (float): The confidence threshold for detection.
    - condition (str): SQL condition to filter images.
    - message (str): Success message to display.
    - limit (int): Maximum number of images to process (default is None).
    """
    conn = sqlite3.connect('images.db')
    c = conn.cursor()
    query = f"SELECT id, path FROM images WHERE {condition}"
    if limit:
        query += f" LIMIT {limit}"
    c.execute(query)
    images = c.fetchall()

    for img_id, img_path in images:
        if os.path.exists(img_path):
            print(f"Processing: {img_path}")

            try:
                detected_objects = detect_objects_func(img_path, model_type=model_type, conf_threshold=conf_threshold)

                # Ensure unique tags and remove duplicates
                tags = {obj.get('name', '') for obj in detected_objects if isinstance(obj, dict) and 'name' in obj}
                tags = [tag for tag in tags if tag]  # Remove empty strings

                # Update the database entry with new tags
                c.execute("UPDATE images SET tags = ? WHERE id = ?", (','.join(tags), img_id))
                print(f"Updated tags for image ID {img_id}: {tags}")

            except Exception as e:
                print(f"Error updating tags for image {img_path}: {e}")
        else:
            print(f"Image not found: {img_path}")

    conn.commit()
    conn.close()
    print(message)