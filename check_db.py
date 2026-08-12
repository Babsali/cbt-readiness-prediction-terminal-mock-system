# check_db.py
import sqlite3

conn = sqlite3.connect("database/cbt_system.db")
cursor = conn.cursor()

cursor.execute("""
    SELECT COUNT(DISTINCT SA.user_id)
    FROM Student_Attempts SA
    JOIN Users U ON SA.user_id = U.user_id
    WHERE U.role = 'Student'
""")
count = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM Student_Attempts")
attempts = cursor.fetchone()[0]

print(f"Total students with attempts: {count}")
print(f"Total attempts in DB: {attempts}")
conn.close()