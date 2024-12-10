import sqlite3

def get_all_images():
    conn = sqlite3.connect('images.db')
    c = conn.cursor()
    c.execute("SELECT path FROM images")
    rows = c.fetchall()
    conn.close()
    return [row[0] for row in rows]

# Get all images from the database
image_paths = get_all_images()

# Print the image paths
print("Images in the database:")
for path in image_paths:
    print(path)