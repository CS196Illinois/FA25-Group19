import sqlite3
import os
from datetime import datetime

# Path to SQLite database file
DB_PATH = os.path.join(os.path.dirname(__file__), "images.db")

# Create tables if not exist
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Table to store top images and their embeddings
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS tops (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT UNIQUE,
        image_data BLOB,
        embedding BLOB,
        uploaded_at TEXT
    )
    """)

    # Table to store bottom images and their embeddings
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS bottoms (
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
        top_id INTEGER,
        bottom_id INTEGER,
        similarity REAL,
        FOREIGN KEY(top_id) REFERENCES tops(id),
        FOREIGN KEY(bottom_id) REFERENCES bottoms(id)
    )
    """)

    # Table to store complete outfits (photo of person wearing full outfit)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS outfits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        outfit_image_data BLOB,
        top_embedding BLOB,
        bottom_embedding BLOB,
        uploaded_at TEXT
    )
    """)

    conn.commit()
    conn.close()


def insert_image(filename, image_data, embedding, category):
    """Insert image into tops or bottoms table based on category"""
    table = "tops" if category == "top" else "bottoms"
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f"""
        INSERT OR REPLACE INTO {table} (filename, image_data, embedding, uploaded_at)
        VALUES (?, ?, ?, ?)
    """, (filename, image_data, bytes(embedding), datetime.now().isoformat()))
    image_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return image_id


def get_all_images(category):
    """Get all images from tops or bottoms table"""
    table = "tops" if category == "top" else "bottoms"
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f"SELECT id, filename, uploaded_at FROM {table}")
    rows = cursor.fetchall()
    conn.close()
    return [{"id": r[0], "filename": r[1], "uploaded_at": r[2]} for r in rows]


def get_image_by_id(image_id, category):
    """Get image data by id from tops or bottoms table"""
    table = "tops" if category == "top" else "bottoms"
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f"SELECT id, filename, image_data, embedding FROM {table} WHERE id = ?", (image_id,))
    row = cursor.fetchone()
    conn.close()
    return row


def get_image_by_filename(filename, category):
    """Get image by filename from tops or bottoms table"""
    table = "tops" if category == "top" else "bottoms"
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f"SELECT id, embedding FROM {table} WHERE filename = ?", (filename,))
    row = cursor.fetchone()
    conn.close()
    return row


def get_all_embeddings(category):
    """Get all embeddings from tops or bottoms table"""
    table = "tops" if category == "top" else "bottoms"
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f"SELECT id, filename, embedding FROM {table}")
    rows = cursor.fetchall()
    conn.close()
    return rows


def delete_image(image_id, category):
    """Delete image from tops or bottoms table"""
    table = "tops" if category == "top" else "bottoms"
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f"DELETE FROM {table} WHERE id = ?", (image_id,))
    conn.commit()
    conn.close()


def insert_similarity(top_id, bottom_id, similarity):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO similarities (top_id, bottom_id, similarity)
        VALUES (?, ?, ?)
    """, (top_id, bottom_id, similarity))
    conn.commit()
    conn.close()


def insert_outfit(outfit_image_data, top_embedding, bottom_embedding):
    """Insert a complete outfit with embeddings for top and bottom regions"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO outfits (outfit_image_data, top_embedding, bottom_embedding, uploaded_at)
        VALUES (?, ?, ?, ?)
    """, (outfit_image_data, bytes(top_embedding), bytes(bottom_embedding), datetime.now().isoformat()))
    outfit_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return outfit_id


def get_all_outfits():
    """Get all outfits with metadata (no image data to keep response light)"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, uploaded_at FROM outfits")
    rows = cursor.fetchall()
    conn.close()
    return [{"id": r[0], "uploaded_at": r[1]} for r in rows]


def get_outfit_by_id(outfit_id):
    """Get complete outfit data including image and embeddings"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, outfit_image_data, top_embedding, bottom_embedding, uploaded_at FROM outfits WHERE id = ?", (outfit_id,))
    row = cursor.fetchone()
    conn.close()
    return row


def get_all_outfit_embeddings():
    """Get all outfit embeddings for similarity search"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, top_embedding, bottom_embedding FROM outfits")
    rows = cursor.fetchall()
    conn.close()
    return rows


def delete_outfit(outfit_id):
    """Delete an outfit by id"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM outfits WHERE id = ?", (outfit_id,))
    conn.commit()
    conn.close()