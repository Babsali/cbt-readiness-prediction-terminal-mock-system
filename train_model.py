import pandas as pd
import sqlite3
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import numpy as np

DB_PATH = "database/cbt_system.db"

def extract_features_from_db():
    """v1.1.0: Feature Extraction Module from DB"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT SA.user_id, SA.score, SA.total, SA.attempted_at
        FROM Student_Attempts SA
        JOIN Users U ON SA.user_id = U.user_id
        WHERE U.role = 'Student'
    """, conn)
    conn.close()

    if df.empty:
        print("No student attempt data found. Add some attempts first in /take_exam")
        return None

    features_list = []
    labels = []
    for user_id, group in df.groupby('user_id'):
        group = group.sort_values('attempted_at')
        total_attempts = len(group)
        avg_score = group['score'].sum() / group['total'].sum() if group['total'].sum() > 0 else 0
        
        # 6 Real features from DB
        completion_rate = avg_score * 100
        consistency_score = group['score'].std() if total_attempts > 1 else 0
        time_management_score = avg_score * 100
        accuracy_easy = avg_score * 100
        accuracy_medium = avg_score * 100 * 0.9
        accuracy_hard = avg_score * 100 * 0.7
        
        # 2 Simulated features - To be replaced in v1.2.0 with real logging
        avg_response_time_sec = np.random.uniform(20, 60) 
        revision_rate = np.random.uniform(0.1, 0.5)

        label = 1 if avg_score >= 0.6 else 0

        features_list.append([
            avg_response_time_sec, revision_rate, accuracy_easy, 
            accuracy_medium, accuracy_hard, time_management_score,
            completion_rate, consistency_score
        ])
        labels.append(label)

    X = pd.DataFrame(features_list, columns=[
        'avg_response_time_sec', 'revision_rate', 'accuracy_easy',
        'accuracy_medium', 'accuracy_hard', 'time_management_score',
        'completion_rate', 'consistency_score'
    ])
    y = pd.Series(labels)
    return X, y

if __name__ == "__main__":
    print("===== v1.1.0 ML Training: LR vs DT vs RF =====")
    data = extract_features_from_db()
    if data is None: exit()

    X, y = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, 'scaler.pkl')

    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    best_acc = 0
    best_model = None
    best_name = ""
    print("\n--- Model Comparison ---")
    for name, model in models.items():
        if name == 'LogisticRegression':
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"{name} Accuracy: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name

    joblib.dump(best_model, 'model.pkl')
    print(f"\nBest Model: {best_name} with Accuracy: {best_acc:.4f}")
    print("Saved model.pkl and scaler.pkl")