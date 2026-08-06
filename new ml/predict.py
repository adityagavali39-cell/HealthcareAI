import pandas as pd
import joblib

# ==========================================
# LOAD MODEL
# ==========================================
model = joblib.load("models/xgboost.pkl")      # किंवा random_forest.pkl

# ==========================================
# LOAD ENCODERS
# ==========================================
gender_encoder = joblib.load("models/gender_encoder.pkl")
bp_encoder = joblib.load("models/bp_encoder.pkl")
chol_encoder = joblib.load("models/chol_encoder.pkl")
disease_encoder = joblib.load("models/disease_encoder.pkl")

# ==========================================
# LOAD FEATURE NAMES
# ==========================================
feature_names = joblib.load("models/feature_names.pkl")

# ==========================================
# CREATE EMPTY INPUT
# ==========================================
input_data = {}

for feature in feature_names:
    input_data[feature] = 0

# ==========================================
# USER DETAILS
# ==========================================
input_data["Age"] = 35
input_data["Gender"] = gender_encoder.transform(["Male"])[0]
input_data["BMI"] = 27.8
input_data["Blood_Pressure"] = bp_encoder.transform(["Normal"])[0]
input_data["Cholesterol_Level"] = chol_encoder.transform(["Normal"])[0]

# ==========================================
# SELECTED SYMPTOMS
# ==========================================
selected_symptoms = [
    "wheezing",
    "shortness of breath",
    "persistent cough",
    "chest tightness",
    "difficulty breathing",
    "fatigue",
    "cough",
    "chest pain"
]

for symptom in selected_symptoms:
    if symptom in input_data:
        input_data[symptom] = 1

# ==========================================
# CONVERT TO DATAFRAME
# ==========================================
input_df = pd.DataFrame([input_data])

# ==========================================
# PREDICTION
# ==========================================
prediction = model.predict(input_df)[0]

probabilities = model.predict_proba(input_df)[0]
confidence = max(probabilities) * 100

predicted_disease = disease_encoder.inverse_transform([prediction])[0]

# ==========================================
# DISPLAY PREDICTION
# ==========================================
print("=" * 60)
print("SMART HEALTHCARE ASSISTANT")
print("=" * 60)

print(f"Predicted Disease : {predicted_disease}")
print(f"Confidence        : {confidence:.2f}%")

print("=" * 60)

# ==========================================
# LOAD RECOMMENDATION DATASET
# ==========================================
recommendations = pd.read_csv("dataset/disease_recommendations.csv")

result = recommendations[
    recommendations["Disease"] == predicted_disease
]

if not result.empty:

    print(f"Severity          : {result.iloc[0]['Severity']}")

    print("\nRecommended Doctor")
    print(result.iloc[0]["Doctor"])

    print("\nDiet")
    print(result.iloc[0]["Diet"])

    print("\nWorkout")
    print(result.iloc[0]["Workout"])

    print("\nPrecautions")
    print(result.iloc[0]["Precautions"])

print("=" * 60)