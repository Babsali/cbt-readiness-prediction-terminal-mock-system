from flask import Flask, render_template, request, redirect, url_for, flash
import sqlite3
import os
from werkzeug.security import generate_password_hash
import joblib
import numpy as np

app = Flask(__name__)
app.secret_key = 'readiness_secret_key_2026'
DB_PATH = "database/cbt_system.db"

# Load ML model once at startup
model = joblib.load('model.pkl')

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def ensure_tables():
    os.makedirs("database", exist_ok=True)
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS Users (
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        full_name TEXT,
        password_hash TEXT,
        role TEXT CHECK(role IN ('Student','Teacher','Admin')),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS Questions (
        question_id INTEGER PRIMARY KEY AUTOINCREMENT,
        question_text TEXT NOT NULL,
        option_a TEXT, option_b TEXT, option_c TEXT, option_d TEXT,
        correct_answer TEXT CHECK(correct_answer IN ('A','B','C','D')),
        difficulty TEXT, topic TEXT, source_flag TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS Exams (
        exam_id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        duration_minutes INTEGER DEFAULT 30,
        total_marks INTEGER DEFAULT 0,
        created_by INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(created_by) REFERENCES Users(user_id))""")
    cur.execute("""CREATE TABLE IF NOT EXISTS Exam_Questions (
        exam_id INTEGER, question_id INTEGER,
        PRIMARY KEY(exam_id, question_id),
        FOREIGN KEY(exam_id) REFERENCES Exams(exam_id),
        FOREIGN KEY(question_id) REFERENCES Questions(question_id))""")
    cur.execute("""CREATE TABLE IF NOT EXISTS Student_Attempts (
        attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER, exam_id INTEGER, score INTEGER, total INTEGER,
        attempted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES Users(user_id),
        FOREIGN KEY(exam_id) REFERENCES Exams(exam_id))""")
    conn.commit()
    conn.close()

ensure_tables()

@app.route("/")
def home():
    conn = get_db()
    users = conn.execute("SELECT COUNT(*) as c FROM Users").fetchone()['c']
    qs = conn.execute("SELECT COUNT(*) as c FROM Questions").fetchone()['c']
    exams = conn.execute("SELECT COUNT(*) as c FROM Exams").fetchone()['c']
    attempts = conn.execute("SELECT COUNT(*) as c FROM Student_Attempts").fetchone()['c']
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    conn.close()
    return render_template("index.html", user_count=users, q_count=qs, exam_count=exams, attempt_count=attempts, tables=tables)

@app.route("/add_user", methods=["GET","POST"])
def add_user():
    if request.method == "POST":
        conn = get_db()
        try:
            conn.execute("INSERT INTO Users (username, full_name, password_hash, role) VALUES (?,?,?,?)",
                         (request.form['username'], request.form['full_name'],
                          generate_password_hash(request.form['password']), request.form['role']))
            conn.commit()
            flash("User added!")
            return redirect("/users")
        except Exception as e:
            flash(f"Error: {e}")
        finally:
            conn.close()
    return render_template("add_user.html")

@app.route("/users")
def users_list():
    conn = get_db()
    users = conn.execute("SELECT * FROM Users ORDER BY user_id DESC").fetchall()
    conn.close()
    return render_template("users.html", users=users)

@app.route("/add_question", methods=["GET","POST"])
def add_question():
    if request.method == "POST":
        conn = get_db()
        conn.execute("""INSERT INTO Questions
        (question_text, option_a, option_b, option_c, option_d, correct_answer, difficulty, topic, source_flag)
        VALUES (?,?,?,?,?,?,?,?,?)""",
        (request.form['question_text'], request.form['option_a'], request.form['option_b'],
         request.form['option_c'], request.form['option_d'], request.form['correct_answer'],
         request.form['difficulty'], request.form['topic'], request.form['source_flag']))
        conn.commit()
        conn.close()
        flash("Question added!")
        return redirect("/questions")
    return render_template("add_question.html")

@app.route("/questions")
def questions_list():
    conn = get_db()
    qs = conn.execute("SELECT * FROM Questions ORDER BY question_id DESC").fetchall()
    conn.close()
    return render_template("questions.html", questions=qs)

@app.route("/add_exam", methods=["GET","POST"])
def add_exam():
    conn = get_db()
    users = conn.execute("SELECT * FROM Users").fetchall()
    if request.method == "POST":
        title = request.form.get('title')
        duration = request.form.get('duration') or 30
        created_by = request.form.get('created_by')
        if not created_by:
            created_by = users[0]['user_id'] if users else None
        conn.execute("INSERT INTO Exams (title, duration_minutes, created_by) VALUES (?,?,?)",
                     (title, duration, created_by))
        conn.commit()
        conn.close()
        return redirect("/exams")
    conn.close()
    return render_template("add_exam.html", users=users)

@app.route("/exams")
def exams_list():
    conn = get_db()
    exams = conn.execute("SELECT Exams.*, Users.username FROM Exams LEFT JOIN Users ON Exams.created_by=Users.user_id ORDER BY exam_id DESC").fetchall()
    conn.close()
    return render_template("exams.html", exams=exams)

@app.route("/exam/<int:exam_id>/manage")
def manage_exam(exam_id):
    conn = get_db()
    exam = conn.execute("SELECT * FROM Exams WHERE exam_id=?", (exam_id,)).fetchone()
    all_qs = conn.execute("SELECT * FROM Questions").fetchall()
    linked = conn.execute("SELECT question_id FROM Exam_Questions WHERE exam_id=?", (exam_id,)).fetchall()
    linked_ids = [r['question_id'] for r in linked]
    conn.close()
    return render_template("manage_exam.html", exam=exam, all_questions=all_qs, linked_ids=linked_ids)

@app.route("/exam/<int:exam_id>/link/<int:q_id>")
def link_question(exam_id, q_id):
    conn = get_db()
    try:
        conn.execute("INSERT INTO Exam_Questions (exam_id, question_id) VALUES (?,?)", (exam_id, q_id))
        conn.commit()
    except:
        pass
    conn.close()
    return redirect(f"/exam/{exam_id}/manage")

@app.route("/exam/<int:exam_id>/unlink/<int:q_id>")
def unlink_question(exam_id, q_id):
    conn = get_db()
    conn.execute("DELETE FROM Exam_Questions WHERE exam_id=? AND question_id=?", (exam_id, q_id))
    conn.commit()
    conn.close()
    return redirect(f"/exam/{exam_id}/manage")

@app.route("/take_exam/<int:exam_id>")
def take_exam(exam_id):
    conn = get_db()
    exam = conn.execute("SELECT * FROM Exams WHERE exam_id=?", (exam_id,)).fetchone()
    questions = conn.execute("""
        SELECT Q.* FROM Questions Q JOIN Exam_Questions EQ ON Q.question_id=EQ.question_id
        WHERE EQ.exam_id=?""", (exam_id,)).fetchall()
    users = conn.execute("SELECT * FROM Users WHERE role='Student'").fetchall()
    if not users:
        users = conn.execute("SELECT * FROM Users").fetchall()
    conn.close()
    return render_template("take_exam.html", exam=exam, questions=questions, users=users)

@app.route("/submit_exam", methods=["POST"])
def submit_exam():
    exam_id = request.form['exam_id']
    user_id = request.form['user_id']
    conn = get_db()
    questions = conn.execute("""
        SELECT Q.* FROM Questions Q JOIN Exam_Questions EQ ON Q.question_id=EQ.question_id
        WHERE EQ.exam_id=?""", (exam_id,)).fetchall()
    score = 0
    for q in questions:
        ans = request.form.get(f"q_{q['question_id']}")
        if ans == q['correct_answer']:
            score += 1
    total = len(questions)
    conn.execute("INSERT INTO Student_Attempts (user_id, exam_id, score, total) VALUES (?,?,?,?)",
                 (user_id, exam_id, score, total))
    conn.commit()
    conn.close()
    return render_template("result.html", score=score, total=total)

@app.route("/attempts")
def attempts_list():
    conn = get_db()
    attempts = conn.execute("""
        SELECT SA.*, U.username, E.title FROM Student_Attempts SA
        JOIN Users U ON SA.user_id=U.user_id
        JOIN Exams E ON SA.exam_id=E.exam_id
        ORDER BY SA.attempt_id DESC
    """).fetchall()
    conn.close()
    return render_template("attempts.html", attempts=attempts)

@app.route("/readiness")
def readiness_dashboard():
    conn = get_db()
    data = conn.execute("""
        SELECT SA.*, U.username, U.full_name, E.title
        FROM Student_Attempts SA
        JOIN Users U ON SA.user_id=U.user_id
        JOIN Exams E ON SA.exam_id=E.exam_id
        ORDER BY SA.attempted_at DESC
    """).fetchall()
    students = conn.execute("SELECT * FROM Users WHERE role='Student'").fetchall()
    if not students:
        students = conn.execute("SELECT * FROM Users").fetchall()
    stats = []
    for s in students:
        attempts = conn.execute("SELECT * FROM Student_Attempts WHERE user_id=?", (s['user_id'],)).fetchall()
        if attempts:
            total_score = sum(a['score'] for a in attempts)
            total_possible = sum(a['total'] for a in attempts)
            avg = (total_score / total_possible * 100) if total_possible > 0 else 0
            if avg >= 70:
                prediction = "HIGHLY READY"
                color = "green"
            elif avg >= 50:
                prediction = "MODERATELY READY"
                color = "orange"
            else:
                prediction = "NOT READY"
                color = "red"
            stats.append({
                'username': s['username'],
                'full_name': s['full_name'],
                'attempts': len(attempts),
                'avg': int(avg),
                'prediction': prediction,
                'color': color
            })
    conn.close()
    return render_template("readiness.html", stats=stats, all_attempts=data)

# ===== ML PREDICTION ROUTE =====
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # 1. Get the 8 features for your thesis from the form
        try:
            features = [
                float(request.form['avg_response_time_sec']),
                float(request.form['revision_rate']),
                float(request.form['accuracy_easy']),
                float(request.form['accuracy_medium']),
                float(request.form['accuracy_hard']),
                float(request.form['time_management_score']),
                float(request.form['completion_rate']),
                float(request.form['consistency_score'])
            ]
        except ValueError:
            flash("Please enter valid numbers for all fields")
            return render_template('predict.html')

        # 2. Predict with the real ML model
        features_np = np.array([features])
        prediction_int = model.predict(features_np)[0] # 0 or 1
        probability = model.predict_proba(features_np)[0][1] # probability of class 1 = Ready

        # 3. Convert to readable output
        if prediction_int == 1:
            prediction = "READY FOR EXAM"
            color = "#0e9f6e"
        else:
            prediction = "NOT READY"
            color = "#dc2626"

        score = round(probability * 100, 2)

        return render_template("predict_result.html",
                               prediction=prediction,
                               color=color,
                               score=score,
                               features=features)

    return render_template('predict.html')

if __name__ == "__main__":
    app.run(debug=True)