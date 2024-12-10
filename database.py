import sqlite3
import numpy as np

def init_db():
    conn = sqlite3.connect('images.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS images
                 (id INTEGER PRIMARY KEY, path TEXT, features BLOB)''')
    conn.commit()
    conn.close()

def get_all_images():
    conn = sqlite3.connect('images.db')
    c = conn.cursor()
    c.execute("SELECT path FROM images")
    rows = c.fetchall()
    conn.close()
    return [row[0] for row in rows]


def save_image_features(img_path, features):
    conn = sqlite3.connect('images.db')
    c = conn.cursor()
    c.execute("INSERT INTO images (path, features) VALUES (?, ?)", (img_path, features.tobytes()))
    conn.commit()
    conn.close()


def query_database(features, uploaded_path, threshold=0.3, debug=True):
    """
    Queries the database to find images similar to the uploaded image,
    excluding any images from the 'uploads' directory.

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
        
        # Exclude images from the 'uploads' directory
        if 'static/uploads' in db_path:
            continue
        
        db_features = np.frombuffer(row[1], dtype=np.float32)

        # Normalize the feature vectors
        db_features = db_features / np.linalg.norm(db_features)
        normalized_features = features / np.linalg.norm(features)

        # Calculate cosine similarity
        similarity = np.dot(normalized_features, db_features)
        distances.append((db_path, similarity))

        # Debugging output
        if debug:
            print(f"Comparing with: {db_path}")
            print(f"Similarity: {similarity:.4f}")

    # Filter images with similarity >= 80% (i.e., similarity >= 0.8)
    similar_images = [(path, score) for path, score in distances if score >= 0.7]
    return similar_images if similar_images else []