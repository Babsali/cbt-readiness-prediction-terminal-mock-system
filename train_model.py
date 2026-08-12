import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

print("Generating dummy training data...")
np.random.seed(42)
n = 500
data = {
    'avg_score': np.random.uniform(40, 100, n),
    'time_spent': np.random.uniform(5, 60, n),
    'login_freq': np.random.randint(1, 20, n),
    'attempts': np.random.randint(1, 5, n),
    'difficulty_pref': np.random.uniform(1, 5, n),
    'topic_coverage': np.random.uniform(20, 100, n),
    'consistency': np.random.uniform(0.3, 1.0, n),
    'last_active_days': np.random.randint(0, 30, n),
}
df = pd.DataFrame(data)
df['ready'] = ((df['avg_score'] > 70) & (df['consistency'] > 0.6)).astype(int)

X = df.drop('ready', axis=1)
y = df['ready']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"Model Accuracy: {model.score(X_test, y_test):.2f}")
joblib.dump(model, 'model.pkl')
print("✅ model.pkl saved!")