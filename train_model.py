import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# 1. Load Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
           "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
df = pd.read_csv(url, names=columns, na_values="?")

# 2. Preprocessing
df = df.fillna(df.median()) # Fill missing values
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0) # 1=Disease, 0=Healthy

# 3. Train AI
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Save the "Brain"
with open('heart_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved as heart_model.pkl")