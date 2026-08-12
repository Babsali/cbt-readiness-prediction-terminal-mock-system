import sqlite3

conn = sqlite3.connect("database/cbt_system.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS Exam_Events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    exam_id INTEGER,
    event_type TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES Users(user_id)
)
""")  

conn.commit()
conn.close()
print("Exam_Events table created successfully!")