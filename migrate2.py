import sqlite3

conn = sqlite3.connect('database/cbt_system.db')
c = conn.cursor()

try:
    c.execute('ALTER TABLE ExamAttempt ADD COLUMN readiness_score REAL DEFAULT 0')
    print('Added readiness_score')
except:
    print('readiness_score already exists')

try:
    c.execute('ALTER TABLE ExamAttempt ADD COLUMN readiness_class TEXT DEFAULT "At-Risk"')
    print('Added readiness_class')
except:
    print('readiness_class already exists')

conn.commit()
conn.close()
print('Done. Now restart app.py')