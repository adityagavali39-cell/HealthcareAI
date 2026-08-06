import joblib
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Load Model
# ----------------------------
model = joblib.load("models/xgboost.pkl")

# ----------------------------
# Load Feature Names
# ----------------------------
feature_names = joblib.load("models/feature_names.pkl")

# ----------------------------
# Feature Importance
# ----------------------------
importance = model.feature_importances_

feature_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

# Sort by importance
feature_df = feature_df.sort_values(
    by="Importance",
    ascending=False
)

print("="*60)
print("TOP 20 IMPORTANT FEATURES")
print("="*60)

print(feature_df.head(20))

# ----------------------------
# Plot Top 20 Features
# ----------------------------
top20 = feature_df.head(20)

plt.figure(figsize=(10,8))
plt.barh(top20["Feature"], top20["Importance"])

plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Top 20 Important Features")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig("models/feature_importance.png", dpi=300)

plt.show()

print("\nGraph saved successfully!")
print("Location : models/feature_importance.png")