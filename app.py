from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import sqlite3
import os
from werkzeug.security import generate_password_hash
import joblib
import numpy as np

app = Flask(__name__)
app.secret_key = 'readiness_secret_key_2026'
DB_PATH = "database/cbt_system.db"

# ===== v1.1.0: LOAD ML MODEL + SCALER =====
# Load the best model trained by train_model.py
model = joblib.load('model.pkl')
# Try to load scaler. Only needed if best model is LogisticRegression
try:
    scaler = joblib.load('scaler.pkl')
    model_name = model.__class__.__name__ # e.g. 'RandomForestClassifier'
except:
    scaler = None
    model_name = model.__class__.__name__

def get_db():
    """Helper to connect to SQLite with Row access"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def ensure_tables():
    """Create all DB tables if they don't exist. Matches Architecture diagram"""
    os.makedirs("database", exist_ok=True)
    conn = get_db()
    cur = conn.cursor()

    # 1. Users table: Students, Teachers, Admins
    cur.execute("""CREATE TABLE IF NOT EXISTS Users (
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        full_name TEXT,
        password_hash TEXT,
        role TEXT CHECK(role IN ('Student','Teacher','Admin')),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")

    # 2. Questions table: Question Bank
    cur.execute("""CREATE TABLE IF NOT EXISTS Questions (
        question_id INTEGER PRIMARY KEY AUTOINCREMENT,
        question_text TEXT NOT NULL,
        option_a TEXT, option_b TEXT, option_c TEXT, option_d TEXT,
        correct_answer TEXT CHECK(correct_answer IN ('A','B','C','D')),
        difficulty TEXT, topic TEXT, source_flag TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")

    # 3. Exams table: CBT Tests
    cur.execute("""CREATE TABLE IF NOT EXISTS Exams (
        exam_id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        duration_minutes INTEGER DEFAULT 30,
        total_marks INTEGER DEFAULT 0,
        created_by INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(created_by) REFERENCES Users(user_id))""")

    # 4. Exam_Questions: Link table for many-to-many
    cur.execute("""CREATE TABLE IF NOT EXISTS Exam_Questions (
        exam_id INTEGER, question_id INTEGER,
        PRIMARY KEY(exam_id, question_id),
        FOREIGN KEY(exam_id) REFERENCES Exams(exam_id),
        FOREIGN KEY(question_id) REFERENCES Questions(question_id))""")

    # 5. Student_Attempts: Behavioral Data Logging Module
    cur.execute("""CREATE TABLE IF NOT EXISTS Student_Attempts (
        attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER, exam_id INTEGER, score INTEGER, total INTEGER,
        attempted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES Users(user_id),
        FOREIGN KEY(exam_id) REFERENCES Exams(exam_id))""")

    # 6. v1.1.0: Predictions table. Log every ML prediction for thesis results
    cur.execute("""CREATE TABLE IF NOT EXISTS Predictions (
        prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        attempt_id INTEGER,
        model_used TEXT,
        prediction_result TEXT,
        confidence REAL,
        predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES Users(user_id),
        FOREIGN KEY(attempt_id) REFERENCES Student_Attempts(attempt_id))""")

    # 7. v1.2.0 NEW: Anti-Cheat Event Logging
    cur.execute("""CREATE TABLE IF NOT EXISTS Exam_Events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        exam_id INTEGER,
        event_type TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES Users(user_id))""")

    conn.commit()
    conn.close()

ensure_tables()

# ========== DASHBOARD & CRUD ROUTES ==========
@app.route("/")
def home():
    """Home Dashboard: Shows counts for all entities"""
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
    """Admin/Teacher can add new users"""
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
    """List all users"""
    conn = get_db()
    users = conn.execute("SELECT * FROM Users ORDER BY user_id DESC").fetchall()
    conn.close()
    return render_template("users.html", users=users)

@app.route("/add_question", methods=["GET","POST"])
def add_question():
    """Add question to Question Bank"""
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
    """List all questions"""
    conn = get_db()
    qs = conn.execute("SELECT * FROM Questions ORDER BY question_id DESC").fetchall()
    conn.close()
    return render_template("questions.html", questions=qs)

@app.route("/add_exam", methods=["GET","POST"])
def add_exam():
    """Create a new Exam/Test"""
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
    """List all exams"""
    conn = get_db()
    exams = conn.execute("SELECT Exams.*, Users.username FROM Exams LEFT JOIN Users ON Exams.created_by=Users.user_id ORDER BY exam_id DESC").fetchall()
    conn.close()
    return render_template("exams.html", exams=exams)

@app.route("/exam/<int:exam_id>/manage")
def manage_exam(exam_id):
    """Link/Unlink questions to an exam"""
    conn = get_db()
    exam = conn.execute("SELECT * FROM Exams WHERE exam_id=?", (exam_id,)).fetchone()
    all_qs = conn.execute("SELECT * FROM Questions").fetchall()
    linked = conn.execute("SELECT question_id FROM Exam_Questions WHERE exam_id=?", (exam_id,)).fetchall()
    linked_ids = [r['question_id'] for r in linked]
    conn.close()
    return render_template("manage_exam.html", exam=exam, all_questions=all_qs, linked_ids=linked_ids)

@app.route("/exam/<int:exam_id>/link/<int:q_id>")
def link_question(exam_id, q_id):
    """Add question to exam"""
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
    """Remove question from exam"""
    conn = get_db()
    conn.execute("DELETE FROM Exam_Questions WHERE exam_id=? AND question_id=?", (exam_id, q_id))
    conn.commit()
    conn.close()
    return redirect(f"/exam/{exam_id}/manage")

@app.route("/take_exam/<int:exam_id>")
def take_exam(exam_id):
    """Student takes the exam. CBT Engine"""
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
    """Auto-grade and log to Student_Attempts"""
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
    """View all student attempts. Raw data for Feature Extraction"""
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
    """T5: Teacher Dashboard. Shows all students and Ready/At-Risk status"""
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
            # Rule-based for now. ML will override this in v1.2.0
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

# ========== v1.1.0: ML PREDICTION ENGINE ==========
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """T6: Student Diagnostics. Runs LR, DT, RF and returns prediction + confidence"""
    if request.method == 'POST':
        # 1. Get the 8 features from the form. These are from Feature Extraction Module
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

        features_np = np.array([features])

        # 2. v1.1.0: Apply StandardScaler if the best model is LogisticRegression
        if model_name == 'LogisticRegression' and scaler is not None:
            features_np = scaler.transform(features_np)

        # 3. Predict with the best ML model loaded from model.pkl
        prediction_int = model.predict(features_np)[0] # 0 = At-Risk, 1 = Ready
        probability = model.predict_proba(features_np)[0][1] # probability of class 1 = Ready
        confidence = round(probability * 100, 2)

        # 4. Convert to readable output for UI
        if prediction_int == 1:
            prediction = "READY"
            color = "#0e9f6e" # green
        else:
            prediction = "AT-RISK"
            color = "#dc2626" # red

        return render_template("predict_result.html",
                               prediction=prediction,
                               color=color,
                               score=confidence, # Confidence %
                               model_used=model_name, # Which of the 3 models won
                               features=features)

    # GET request: just show the form
    return render_template('predict.html')

# ========== v1.2.0: ANTI-CHEAT LOGGING API ==========
@app.route('/api/log_event', methods=['POST'])
def log_event():
    data = request.get_json()

    # Get user_id from session. For now we use 1. Later we use session['user_id']
    user_id = session.get('user_id', 1)
    exam_id = data.get('exam_id')
    event_type = data.get('event_type')

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO Exam_Events (user_id, exam_id, event_type)
        VALUES (?,?,?)
    """, (user_id, exam_id, event_type))

    conn.commit()
    conn.close()

    return jsonify({"status": "logged"}), 200

# ========== v1.2.0: ADMIN VIOLATIONS VIEW ==========
@app.route("/admin/violations/<int:exam_id>")
def violations(exam_id):
    """View all anti-cheat events for an exam"""
    conn = get_db()
    events = conn.execute("""
        SELECT EE.*, U.username, U.full_name FROM Exam_Events EE
        JOIN Users U ON EE.user_id=U.user_id
        WHERE EE.exam_id=? ORDER BY EE.timestamp DESC
    """, (exam_id,)).fetchall()
    exam = conn.execute("SELECT * FROM Exams WHERE exam_id=?", (exam_id,)).fetchone()
    conn.close()
    return render_template("violations.html", events=events, exam=exam)

if __name__ == "__main__":
    app.run(debug=True)