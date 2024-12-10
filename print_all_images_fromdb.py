import sqlite3

def print_all_images():
    conn = sqlite3.connect('images.db')
    c = conn.cursor()
    c.execute("SELECT path FROM images")
    rows = c.fetchall()
    conn.close()
    for row in rows:
        print(row[0])

print_all_images()