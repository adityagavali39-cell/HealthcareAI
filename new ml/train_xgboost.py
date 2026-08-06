import pandas as pd
import joblib

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("new ml/dataset/processed_dataset.csv")

# ----------------------------
# Features & Target
# ----------------------------
X = df.drop("Disease", axis=1)
y = df["Disease"]

# ----------------------------
# Train Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# ----------------------------
# Base Model
# ----------------------------
xgb = XGBClassifier(
    objective="multi:softprob",
    num_class=len(y.unique()),
    eval_metric="mlogloss",
    random_state=42
)

# ----------------------------
# Hyperparameter Grid
# ----------------------------
params = {

    "n_estimators": [100, 200, 300, 500],

    "max_depth": [3, 5, 7, 9],

    "learning_rate": [0.01, 0.05, 0.1],

    "subsample": [0.8, 0.9, 1.0],

    "colsample_bytree": [0.8, 0.9, 1.0],

    "min_child_weight": [1, 3, 5],

    "gamma": [0, 0.1, 0.2]
}

# ----------------------------
# Random Search
# ----------------------------
search = RandomizedSearchCV(

    estimator=xgb,

    param_distributions=params,

    n_iter=20,

    cv=5,

    scoring="accuracy",

    verbose=2,

    random_state=42,

    n_jobs=-1

)

print("\nSearching Best Parameters...\n")

search.fit(X_train, y_train)

# ----------------------------
# Best Model
# ----------------------------
best_model = search.best_estimator_

print("\nBest Parameters\n")
print(search.best_params_)

# ----------------------------
# Prediction
# ----------------------------
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nFinal Accuracy :", round(accuracy*100,2), "%")

print(classification_report(y_test, y_pred))

# ----------------------------
# Save Model
# ----------------------------
import os

os.makedirs("new ml/models", exist_ok=True)

joblib.dump(best_model, "new ml/models/xgboost.pkl")

print("\nModel Saved Successfully")