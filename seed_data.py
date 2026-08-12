# seed_data.py
import sqlite3
import random
from datetime import datetime, timedelta

DB_PATH = "database/cbt_system.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

print("Seeding 10 fake students...")

for i in range(1, 11): # 10 students
    username = f"student{i}"
    # 1. Add user if not exist
    cursor.execute("INSERT OR IGNORE INTO Users (username, password_hash, role) VALUES (?,?,?)",
                   (username, "pbkdf2:sha256:fakehash", "Student"))

    # 2. Get user_id
    cursor.execute("SELECT user_id FROM Users WHERE username =?", (username,))
    user_id = cursor.fetchone()[0]

    # 3. Give each student 2-4 attempts with different scores
    for j in range(random.randint(2,4)):
        score = random.randint(30, 95) # some pass, some fail
        total = 100
        date = datetime.now() - timedelta(days=j+1)
        cursor.execute("INSERT INTO Student_Attempts (user_id, exam_id, score, total, attempted_at) VALUES (?,?,?,?,?)",
                       (user_id, 1, score, total, date))

conn.commit()
conn.close()
print("Done! 10 students with 2-4 attempts each added.")