import sqlite3
conn = sqlite3.connect("database/cbt_system.db")
try:
    conn.execute("ALTER TABLE Questions ADD COLUMN image_path TEXT")
    print("Column image_path added successfully")
except sqlite3.OperationalError as e:
    print("Column probably already exists:", e)
conn.commit()
conn.close()