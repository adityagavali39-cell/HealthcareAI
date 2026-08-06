import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier

# -----------------------
# Load Dataset
# -----------------------
df = pd.read_csv("new ml/dataset/processed_dataset.csv")

X = df.drop("Disease", axis=1)
y = df["Disease"]

# -----------------------
# Split Dataset
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# -----------------------
# Random Forest
# -----------------------
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# -----------------------
# Tuned XGBoost
# -----------------------
xgb = XGBClassifier(
    objective="multi:softprob",
    num_class=len(y.unique()),
    n_estimators=200,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=1.0,
    gamma=0.2,
    min_child_weight=5,
    eval_metric="mlogloss",
    random_state=42
)

# -----------------------
# Stacking
# -----------------------
stack_model = StackingClassifier(
    estimators=[
        ("rf", rf),
        ("xgb", xgb)
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
    n_jobs=-1
)

print("\nTraining Stacking Model...")

stack_model.fit(X_train, y_train)

# -----------------------
# Prediction
# -----------------------
y_pred = stack_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy :", round(accuracy*100,2), "%")

print(classification_report(y_test,y_pred))

# -----------------------
# Save Model
# -----------------------
joblib.dump(stack_model,"new ml/models/stacking.pkl")

print("\nStacking Model Saved Successfully!")