from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, Response

import sqlite3
import os
import csv
from io import TextIOWrapper
from werkzeug.security import generate_password_hash
import joblib
import numpy as np

# ===== NEW IMPORTS FOR v1.3.0-ui-upload =====
import pandas as pd
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import io
import re
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'readiness_secret_key_2026'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16MB limit
DB_PATH = "database/cbt_system.db"
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ===== IMPORTANT: SET PATHS FOR WINDOWS =====
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ["PATH"] += os.pathsep + r'C:\poppler-26.02.0\Library\bin'

# ===== v1.1.0: LOAD ML MODEL + SCALER =====
model = joblib.load('model.pkl')
try:
    scaler = joblib.load('scaler.pkl')
    model_name = model.__class__.__name__
except:
    scaler = None
    model_name = model.__class__.__name__

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def ensure_tables():
    os.makedirs("database", exist_ok=True)
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS Users (user_id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, full_name TEXT, password_hash TEXT, role TEXT CHECK(role IN ('Student','Teacher','Admin')), created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS Questions (question_id INTEGER PRIMARY KEY AUTOINCREMENT, question_text TEXT NOT NULL, option_a TEXT, option_b TEXT, option_c TEXT, option_d TEXT, correct_answer TEXT CHECK(correct_answer IN ('A','B','C','D')), difficulty TEXT, topic TEXT, source_flag TEXT, image_path TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS Exams (exam_id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, duration_minutes INTEGER DEFAULT 30, total_marks INTEGER DEFAULT 0, created_by INTEGER, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY(created_by) REFERENCES Users(user_id))""")
    cur.execute("""CREATE TABLE IF NOT EXISTS Exam_Questions (exam_id INTEGER, question_id INTEGER, PRIMARY KEY(exam_id, question_id), FOREIGN KEY(exam_id) REFERENCES Exams(exam_id), FOREIGN KEY(question_id) REFERENCES Questions(question_id))""")
    cur.execute("""CREATE TABLE IF NOT EXISTS Student_Attempts (attempt_id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, exam_id INTEGER, score INTEGER, total INTEGER, attempted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY(user_id) REFERENCES Users(user_id), FOREIGN KEY(exam_id) REFERENCES Exams(exam_id))""")
    cur.execute("""CREATE TABLE IF NOT EXISTS Predictions (prediction_id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, attempt_id INTEGER, model_used TEXT, prediction_result TEXT, confidence REAL, predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY(user_id) REFERENCES Users(user_id), FOREIGN KEY(attempt_id) REFERENCES Student_Attempts(attempt_id))""")
    cur.execute("""CREATE TABLE IF NOT EXISTS Exam_Events (event_id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, exam_id INTEGER, event_type TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY(user_id) REFERENCES Users(user_id))""")
    conn.commit()
    conn.close()

ensure_tables()

def smart_parse_from_file(file, ext):
    file.stream.seek(0)
    text = ""
    try:
        if ext == 'pdf':
            images = convert_from_bytes(file.read(), poppler_path=r'C:\poppler-26.02.0\Library\bin')
            for img in images:
                text += pytesseract.image_to_string(img, lang='eng') + "\n"
        else:
            img = Image.open(file.stream)
            img = img.convert('L').resize((img.width*2, img.height*2))
            text = pytesseract.image_to_string(img, lang='eng', config='--psm 6')
    except Exception as e:
        print("OCR ERROR:", e)
        raise e
    return parse_text_to_questions(text)

def parse_text_to_questions(text):
    questions = []
    text = text.replace('Vurce', 'A.').replace('formulk', 'formula').replace('wrotted', 'wrote')
    raw_blocks = re.split(r'(?=\n\s*\.\s*[^A-D]|Answer:\s*[A-D])', text, flags=re.I)
    current = {}
    for block in raw_blocks:
        block = block.strip()
        if not block: continue
        q_match = re.search(r'([^\n\?]+\?)', block)
        if q_match and 'question_text' not in current:
            current['question_text'] = auto_latex(q_match.group(1).strip())
        opts = re.findall(r'[A-D]\.?\s*([^\n]+)', block)
        opts = [o.strip() for o in opts if len(o.strip()) > 1]
        if len(opts) >= 2:
            current['option_a'] = auto_latex(opts[0]) if len(opts) > 0 else ''
            current['option_b'] = auto_latex(opts[1]) if len(opts) > 1 else ''
            current['option_c'] = auto_latex(opts[2]) if len(opts) > 2 else ''
            current['option_d'] = auto_latex(opts[3]) if len(opts) > 3 else ''
        ans_match = re.search(r'[Aa]nswer[:\s]*([A-D1-4])', block)
        if ans_match:
            ans = ans_match.group(1).upper()
            if ans.isdigit(): ans = ['A','B','C','D'][int(ans)-1]
            current['correct_answer'] = ans
            current['difficulty'] = rate_difficulty(current.get('question_text',''))
            current['topic'] = 'General'
            if 'question_text' in current and 'correct_answer' in current:
                questions.append(current)
                current = {}
    return questions

def auto_latex(text):
    if any(x in text for x in ['^', '/', 'm/s', 'cm', '=','+','-']):
        return f"$$ {text} $$"
    return text

def rate_difficulty(text):
    word_count = len(text.split())
    if word_count < 15: return "Easy"
    elif word_count < 30: return "Medium"
    else: return "Hard"

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

@app.route('/admin/dashboard')
def admin_dashboard():
    conn = get_db()
    user_count = conn.execute("SELECT COUNT(*) as c FROM Users").fetchone()['c']
    q_count = conn.execute("SELECT COUNT(*) as c FROM Questions").fetchone()['c']
    exam_count = conn.execute("SELECT COUNT(*) as c FROM Exams").fetchone()['c']
    attempt_count = conn.execute("SELECT COUNT(*) as c FROM Student_Attempts").fetchone()['c']
    conn.close()
    return render_template('admin_dashboard.html', user_count=user_count, q_count=q_count, exam_count=exam_count, attempt_count=attempt_count)

@app.route('/admin/download_template')
def download_template():
    df = pd.DataFrame([{
        'question_text': 'What is the capital of Nigeria?',
        'option_a': 'Lagos', 'option_b': 'Abuja', 'option_c': 'Kano', 'option_d': 'Ibadan',
        'correct_answer': 'B', 'difficulty': 'Easy', 'topic': 'Geography'
    }])
    csv_data = df.to_csv(index=False, encoding='utf-8')
    return Response(csv_data, mimetype="text/csv", headers={"Content-disposition": "attachment; filename=question_template.csv"})

@app.route('/admin/upload_questions', methods=["GET","POST"])
def upload_questions():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'danger')
            return redirect(request.url)
        ext = file.filename.rsplit('.', 1)[1].lower()
        questions = []
        try:
            if ext in ['csv', 'xlsx', 'xls']:
                df = pd.read_csv(file) if ext == 'csv' else pd.read_excel(file)
                questions = df.to_dict('records')
                flash(f'Loaded {len(questions)} questions from {ext.upper()}', 'success')
            elif ext in ['pdf', 'jpg', 'jpeg', 'png']:
                questions = smart_parse_from_file(file, ext)
                flash(f'Parsed {len(questions)} questions from scanned file. Please review.', 'warning')
            else:
                flash(f'Unsupported file type:.{ext}', 'danger')
                return redirect(request.url)
        except Exception as e:
            flash(f'Error parsing file: {str(e)}', 'danger')
            return redirect(request.url)
        return render_template('review_upload.html', questions=questions)
    return render_template('upload_questions.html')

# ===== FIXED CSV UPLOAD - ONLY 1 COPY NOW =====
@app.route('/admin/upload_questions_csv', methods=['GET', 'POST'])
def upload_questions_csv():
    if request.method == 'POST':
        if 'csv_file' not in request.files:
            flash('No file selected', 'danger')
            return redirect(request.url)
        file = request.files['csv_file']
        if file.filename == '':
            flash('No file selected', 'danger')
            return redirect(request.url)
        if file and file.filename.endswith('.csv'):
            try:
                df = pd.read_csv(file)
                conn = get_db()
                count = 0
                for _, row in df.iterrows():
                    conn.execute("""INSERT INTO Questions
                    (question_text, option_a, option_b, option_c, option_d, correct_answer, difficulty, topic, source_flag)
                    VALUES (?,?,?,?,?,?,?,?,?)""", # FIXED: 9 placeholders
                    (row['question_text'], row['option_a'], row['option_b'], row['option_c'], row['option_d'],
                     row['correct_answer'], row.get('difficulty','Medium'), row.get('topic','General'), 'CSV_UPLOAD'))
                    count += 1
                conn.commit()
                conn.close()
                flash(f'Success! {count} questions uploaded.', 'success')
                return redirect(url_for('questions_list'))
            except Exception as e:
                flash(f'Error reading CSV: {e}', 'danger')
                print("CSV ERROR:", e)
    return render_template('admin_upload_csv.html')

@app.route('/admin/save_reviewed_questions', methods=['POST'])
def save_reviewed_questions():
    conn = get_db()
    count = 0
    i = 0
    while f'q_text_{i}' in request.form:
        try:
            conn.execute("""INSERT INTO Questions
            (question_text, option_a, option_b, option_c, option_d, correct_answer, difficulty, topic, source_flag)
            VALUES (?,?,?,?,?,?,?,?,?)""",
            (request.form[f'q_text_{i}'], request.form[f'q_a_{i}'], request.form[f'q_b_{i}'],
             request.form[f'q_c_{i}'], request.form[f'q_d_{i}'], request.form[f'q_ans_{i}'],
             request.form[f'q_diff_{i}'], request.form[f'q_topic_{i}'], 'SMART_UPLOAD'))
            count += 1
        except Exception as e:
            print(f"Error saving question {i}: {e}")
        i += 1
    conn.commit()
    conn.close()
    flash(f'Success! {count} questions saved to question bank.', 'success')
    return redirect('/questions')

@app.route("/add_user", methods=["GET","POST"])
def add_user():
    if request.method == "POST":
        conn = get_db()
        try:
            conn.execute("INSERT INTO Users (username, full_name, password_hash, role) VALUES (?,?,?,?)",
                         (request.form['username'], request.form['full_name'], generate_password_hash(request.form['password']), request.form['role']))
            conn.commit()
            flash("User added!", 'success')
            return redirect("/users")
        except Exception as e:
            flash(f"Error: {e}", 'danger')
        finally:
            conn.close()
    return render_template("add_user.html")

@app.route('/admin/delete_question/<int:question_id>')
def delete_question(question_id):
    db = get_db()
    db.execute("DELETE FROM Questions WHERE question_id =?", (question_id,))
    db.commit()
    return redirect("/questions")

@app.route('/questions')
def questions_list():
    db = get_db()
    qs = db.execute("SELECT * FROM Questions ORDER BY question_id DESC").fetchall()
    db.close()
    return render_template("questions.html", questions=qs)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)