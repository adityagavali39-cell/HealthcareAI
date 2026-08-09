"""
helper.py
==========================================
Knowledge-base lookups used by app.py.

Your "new ml" folder consolidates everything into ONE file:
new ml/dataset/disease_recommendations.csv
with columns: Disease, Diet, Precautions, Severity, Workout, Doctor

NOTE: There is no "Medicines" column in this dataset (the old
ml/knowledge/medicines.csv from your first version isn't part of
"new ml"). get_medicines() below returns a generic safety message
instead of a real lookup -- see the note at the bottom of this file
for how to add real medicine data back in.
"""

import pandas as pd

RECOMMENDATIONS_PATH = "new ml/dataset/disease_recommendations.csv"

recommendations_df = pd.read_csv(RECOMMENDATIONS_PATH)


def _get_row(disease):
    row = recommendations_df[recommendations_df["Disease"] == disease]
    return row.iloc[0] if not row.empty else None


def get_precautions(disease):
    row = _get_row(disease)
    if row is None:
        return ["No precautions available."]
    return [tip.strip() for tip in row["Precautions"].split(";")]


def get_medicines(disease):
    # No Medicines column in disease_recommendations.csv -- see note above.
    return ["Consult your doctor for prescribed medicines."]


def get_doctor(disease):
    row = _get_row(disease)
    if row is None:
        return "No doctor recommendation available."
    return row["Doctor"]


def get_severity(disease):
    row = _get_row(disease)
    if row is None:
        return "Unknown"
    return row["Severity"]


def get_diet(disease):
    row = _get_row(disease)
    if row is None:
        return ["No diet information available."]
    return [food.strip() for food in row["Diet"].split(";")]


def get_workout(disease):
    row = _get_row(disease)
    if row is None:
        return ["No workout information available."]
    return [w.strip() for w in row["Workout"].split(";")]


# ==========================================
# To bring real medicine data back:
# 1. Add a "Medicines" column to
#    new ml/dataset/disease_recommendations.csv
#    (same "item;item;item" format as Diet/Precautions/Workout), or
# 2. Keep a separate medicines.csv (Disease,Medicines) and read it
#    here the same way get_diet() reads Diet.
# ==========================================