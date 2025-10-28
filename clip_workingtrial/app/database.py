import sqlite3
import os
from datetime import datetime

# Path to SQLite database file
DB_PATH = os.path.join(os.path.dirname(__file__), "images.db")

# Create tables if not exist
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Table to store images and their embeddings
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT UNIQUE,
        image_data BLOB,
        embedding BLOB,
        uploaded_at TEXT
    )
    """)

    # Table to store similarity scores between pairs of images
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS similarities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image1_id INTEGER,
        image2_id INTEGER,
        similarity REAL,
        FOREIGN KEY(image1_id) REFERENCES images(id),
        FOREIGN KEY(image2_id) REFERENCES images(id)
    )
    """)

    conn.commit()
    conn.close()


def insert_image(filename, image_data, embedding):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO images (filename, image_data, embedding, uploaded_at)
        VALUES (?, ?, ?, ?)
    """, (filename, image_data, bytes(embedding), datetime.now().isoformat()))
    conn.commit()
    conn.close()


def get_all_images():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, filename, uploaded_at FROM images")
    rows = cursor.fetchall()
    conn.close()
    return [{"id": r[0], "filename": r[1], "uploaded_at": r[2]} for r in rows]


def get_image_by_filename(filename):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, embedding FROM images WHERE filename = ?", (filename,))
    row = cursor.fetchone()
    conn.close()
    return row


def get_all_embeddings():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, filename, embedding FROM images")
    rows = cursor.fetchall()
    conn.close()
    return rows


def insert_similarity(image1_id, image2_id, similarity):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO similarities (image1_id, image2_id, similarity)
        VALUES (?, ?, ?)
    """, (image1_id, image2_id, similarity))
    conn.commit()
    conn.close()
