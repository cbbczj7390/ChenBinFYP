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

def query_database(features, uploaded_path, threshold=0.1):
    conn = sqlite3.connect('images.db')
    c = conn.cursor()
    # Only select paths from the initial_images directory
    c.execute("SELECT path, features FROM images WHERE path LIKE 'static/initial_images/%'")
    rows = c.fetchall()
    conn.close()

    distances = []
    for row in rows:
        db_path = row[0]
        db_features = np.frombuffer(row[1], dtype=np.float32)
        db_features = db_features / np.linalg.norm(db_features)
        similarity = np.dot(features, db_features)  # Cosine similarity
        print(f"Comparing with: {db_path}, Similarity: {similarity:.4f}")  # Debugging print
        distances.append((db_path, similarity))

    # Filter images based on a threshold
    similar_images = [(path, similarity) for path, similarity in distances if similarity >= (1 - threshold)]
    return similar_images if similar_images else []