import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("dataset/final_healthcare_dataset (1).csv")

print("Original Shape:", df.shape)

# ----------------------------
# Remove Duplicate Rows
# ----------------------------
df = df.drop_duplicates()

print("After Removing Duplicates:", df.shape)

# ----------------------------
# Encode Gender
# ----------------------------
gender_encoder = LabelEncoder()
df["Gender"] = gender_encoder.fit_transform(df["Gender"])

# ----------------------------
# Encode Blood Pressure
# ----------------------------
bp_encoder = LabelEncoder()
df["Blood_Pressure"] = bp_encoder.fit_transform(df["Blood_Pressure"])

# ----------------------------
# Encode Cholesterol
# ----------------------------
chol_encoder = LabelEncoder()
df["Cholesterol_Level"] = chol_encoder.fit_transform(df["Cholesterol_Level"])

# ----------------------------
# Encode Disease
# ----------------------------
disease_encoder = LabelEncoder()
df["Disease"] = disease_encoder.fit_transform(df["Disease"])

# ----------------------------
# Create Models Folder
# ----------------------------
os.makedirs("models", exist_ok=True)

# ----------------------------
# Save Encoders
# ----------------------------
joblib.dump(gender_encoder, "models/gender_encoder.pkl")
joblib.dump(bp_encoder, "models/bp_encoder.pkl")
joblib.dump(chol_encoder, "models/chol_encoder.pkl")
joblib.dump(disease_encoder, "models/disease_encoder.pkl")

# ----------------------------
# Save Processed Dataset
# ----------------------------
df.to_csv("dataset/processed_dataset.csv", index=False)

print("\nPreprocessing Completed Successfully!")
print("Processed Dataset Shape:", df.shape)
print("Encoders Saved Successfully!")