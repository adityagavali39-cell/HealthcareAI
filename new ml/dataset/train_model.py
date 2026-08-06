import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ----------------------------
# Load Processed Dataset
# ----------------------------
df = pd.read_csv("dataset/processed_dataset.csv")

# ----------------------------
# Features and Target
# ----------------------------
X = df.drop("Disease", axis=1)
y = df["Disease"]

# ----------------------------
# Save Feature Names
# ----------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(X.columns.tolist(), "models/feature_names.pkl")

# ----------------------------
# Train Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# ----------------------------
# Random Forest Model
# ----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest...")
model.fit(X_train, y_train)

# ----------------------------
# Prediction
# ----------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy :", round(accuracy * 100, 2), "%")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ----------------------------
# Save Model
# ----------------------------
joblib.dump(model, "models/random_forest.pkl")

print("\nModel Saved Successfully!")