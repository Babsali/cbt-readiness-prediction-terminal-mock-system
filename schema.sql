
-- ============================================
-- ReadinessPredictionSystem - Single Tenant Offline CBT
-- SQLite Schema - 7 Tables - No tenant_id (Proof of single-tenant)
-- ============================================

PRAGMA foreign_keys = ON;

-- 1. User Table (Admin, Teacher, Student) - Single Institution Only
CREATE TABLE IF NOT EXISTS User (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('Admin', 'Teacher', 'Student')),
    full_name TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 2. Question Table - Dual Source Flag
CREATE TABLE IF NOT EXISTS Question (
    question_id INTEGER PRIMARY KEY AUTOINCREMENT,
    question_text TEXT NOT NULL,
    option_a TEXT NOT NULL,
    option_b TEXT NOT NULL,
    option_c TEXT NOT NULL,
    option_d TEXT NOT NULL,
    correct_answer TEXT NOT NULL CHECK(correct_answer IN ('A','B','C','D')),
    topic TEXT NOT NULL,
    difficulty TEXT NOT NULL CHECK(difficulty IN ('Easy','Medium','Hard')),
    source_flag TEXT NOT NULL DEFAULT 'Manual' CHECK(source_flag IN ('Manual','AI-Generated')),
    review_status TEXT NOT NULL DEFAULT 'Approved' CHECK(review_status IN ('Pending Review','Approved')),
    created_by INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (created_by) REFERENCES User(user_id) ON DELETE SET NULL
);

-- 3. Exam Table
CREATE TABLE IF NOT EXISTS Exam (
    exam_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    subject TEXT NOT NULL,
    duration_minutes INTEGER NOT NULL,
    total_questions INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'Draft' CHECK(status IN ('Draft','Published','Archived')),
    created_by INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (created_by) REFERENCES User(user_id) ON DELETE SET NULL
);

-- 4. Junction Table - Exam_Question (Many-to-Many)
CREATE TABLE IF NOT EXISTS Exam_Question (
    exam_id INTEGER NOT NULL,
    question_id INTEGER NOT NULL,
    question_order INTEGER NOT NULL,
    PRIMARY KEY (exam_id, question_id),
    FOREIGN KEY (exam_id) REFERENCES Exam(exam_id) ON DELETE CASCADE,
    FOREIGN KEY (question_id) REFERENCES Question(question_id) ON DELETE CASCADE
);

-- 5. Student_Attempt - Logs each submission
CREATE TABLE IF NOT EXISTS Student_Attempt (
    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    exam_id INTEGER NOT NULL,
    raw_score REAL NOT NULL,
    total_score REAL NOT NULL,
    percentage REAL NOT NULL,
    submitted_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES User(user_id) ON DELETE CASCADE,
    FOREIGN KEY (exam_id) REFERENCES Exam(exam_id) ON DELETE CASCADE
);

-- 6. Feature_Log - Stores 12 Features (Table 3.1)
CREATE TABLE IF NOT EXISTS Feature_Log (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    attempt_id INTEGER NOT NULL UNIQUE,
    prior_attempt_score REAL,
    avg_mock_score REAL,
    mock_score_trend REAL,
    avg_response_latency REAL,
    total_session_time INTEGER,
    question_revisions INTEGER,
    easy_accuracy REAL,
    medium_accuracy REAL,
    hard_accuracy REAL,
    mock_completion_count INTEGER,
    topic_score_variance REAL,
    item_difficulty_mix TEXT,
    FOREIGN KEY (attempt_id) REFERENCES Student_Attempt(attempt_id) ON DELETE CASCADE
);

-- 7. Prediction - ML Output
CREATE TABLE IF NOT EXISTS Prediction (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    attempt_id INTEGER NOT NULL UNIQUE,
    predicted_label INTEGER NOT NULL CHECK(predicted_label IN (0,1)), -- 0=At-Risk, 1=Ready
    confidence_score REAL,
    diagnostic_text TEXT, -- e.g. "Weak in Hard Questions, Topic: Algebra"
    prediction_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (attempt_id) REFERENCES Student_Attempt(attempt_id) ON DELETE CASCADE
);
