# database.py
import sqlite3
import numpy as np

def init_db():
    conn = sqlite3.connect('images.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS images
                 (id INTEGER PRIMARY KEY, path TEXT, features BLOB)''')
    conn.commit()
    conn.close()

def save_image_features(img_path, features):
    conn = sqlite3.connect('images.db')
    c = conn.cursor()
    c.execute("INSERT INTO images (path, features) VALUES (?, ?)", (img_path, features.tobytes()))
    conn.commit()
    conn.close()

def query_database(features):
    conn = sqlite3.connect('images.db')
    c = conn.cursor()
    c.execute("SELECT path, features FROM images")
    rows = c.fetchall()
    conn.close()

    distances = []
    for row in rows:
        db_features = np.frombuffer(row[1], dtype=np.float32)
        distance = np.linalg.norm(features - db_features)
        distances.append((row[0], distance))

    distances.sort(key=lambda x: x[1])
    similar_images = [x[0] for x in distances[:3]]
    return similar_images

def get_all_images():
    conn = sqlite3.connect('images.db')
    c = conn.cursor()
    c.execute("SELECT path FROM images")
    rows = c.fetchall()
    conn.close()
    return [row[0] for row in rows]