from flask import Flask, render_template, request, redirect, session, url_for,flash, send_file
from werkzeug.middleware.proxy_fix import ProxyFix
import io
import socket
import requests
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download
import psycopg2
from dotenv import load_dotenv
from database.db import get_connection
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import os
from werkzeug.security import generate_password_hash
import time
from datetime import datetime
from dotenv import load_dotenv
from flask_mail import Mail, Message
load_dotenv()
import random
from werkzeug.security import check_password_hash
from helper import (
    get_precautions,
    get_medicines,
    get_doctor,
    get_diet,
    get_workout,
    get_severity
)
from authlib.integrations.flask_client import OAuth
from dotenv import load_dotenv
import os
load_dotenv()
print("MAIL_SERVER:", os.getenv("MAIL_SERVER"))
print("MAIL_PORT:", os.getenv("MAIL_PORT"))
print("MAIL_USERNAME:", os.getenv("MAIL_USERNAME"))
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
app.secret_key = os.getenv("SECRET_KEY")

# ==========================================
# Email Configuration - Resend
# ==========================================

# Resend uses HTTPS API, so SMTP settings are not required.
RESEND_API_KEY = os.getenv("RESEND_API_KEY")
RESEND_FROM_EMAIL = os.getenv("RESEND_FROM_EMAIL", "onboarding@resend.dev")

if not RESEND_API_KEY:
    print("WARNING: RESEND_API_KEY is not set!")

print("RESEND_FROM_EMAIL:", RESEND_FROM_EMAIL)

# Flask-Mail Message is still used to keep the existing email code simple.
mail = Mail(app)
oauth = OAuth(app)


# ==========================================
# Email Sending Configuration
# ==========================================

def safe_send_mail(msg):
    """
    Send an email using the Resend HTTPS API.

    This avoids direct SMTP connections, which can be blocked on Railway.
    The existing Flask-Mail Message object is accepted so the rest of the
    application does not need to change.
    """
    try:
        if not RESEND_API_KEY:
            print("ERROR: RESEND_API_KEY is missing.")
            return False

        response = requests.post(
         "https://api.resend.com/emails",
          headers={
         "Authorization": f"Bearer {RESEND_API_KEY}",
         "Content-Type": "application/json"
          },
          json={
          "from": RESEND_FROM_EMAIL,
          "to": msg.recipients,
          "subject": msg.subject,
         "text": msg.body
          },
          timeout=15
      )

        print("Resend status:", response.status_code)
        print("Resend response:", response.text)

        if response.status_code in (200, 201):
            print("Email sent successfully to:", msg.recipients)
            return True

        print("Resend email failed:", response.status_code, response.text)
        return False

    except requests.exceptions.Timeout:
        print("Resend request timed out.")
        return False

    except requests.exceptions.RequestException as e:
        print("Resend request error:", e)
        return False

    except Exception as e:
        print("Email sending error:", e)
        return False


# ==========================================
# Google OAuth Configuration
# ==========================================

google = oauth.register(
    name="google",
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={
        "scope": "openid email profile"
    }
)


# ==========================================
# Flask Secret Key
# ==========================================

app.secret_key = os.getenv("SECRET_KEY")


# ==========================================
# Load ML Models
# ==========================================
# Your "new ml" folder is the real, matching set of artifacts:
#   - Random Forest -> new ml/models/random_forest.pkl
#   - XGBoost        -> new ml/models/xgboost.pkl   (the 2 models)
# trained on 119 features: Age, Gender, BMI, Blood_Pressure,
# Cholesterol_Level + 114 binary symptoms (new ml/models/feature_names.pkl).
#
# Requires: pip install xgboost  (needed to unpickle xgboost.pkl,
# even though app.py itself never imports xgboost directly).

# ==========================================
# Hugging Face ML Models
# ==========================================


REPO_ID = "aditya12g/healthcareaimodels"

def load_model(filename):
    print(f"Downloading {filename}...")

    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename
    )

    print(f"Downloaded: {model_path}")

    model = joblib.load(model_path)

    print(f"{filename} loaded successfully")

    return model


#       rf_model = load_model("random_forest.pkl")
stack_model = load_model("stacking.pkl")
# xgb_model = load_model("xgboost.pkl")

feature_names = load_model("feature_names.pkl")
disease_encoder = load_model("disease_encoder.pkl")
gender_encoder = load_model("gender_encoder.pkl")
bp_encoder = load_model("bp_encoder.pkl")
chol_encoder = load_model("chol_encoder.pkl")

NON_SYMPTOM_FEATURES = ["Age", "Gender", "BMI", "Blood_Pressure", "Cholesterol_Level"]


def encode_gender(gender):
    """
    gender_encoder was trained on a dataset that only contains
    Male/Female (see: gender_encoder.classes_). predict.html still
    offers an "Other" option, which the encoder has never seen, so
    we fall back to "Female" for anything it doesn't recognize.
    To fix properly: retrain preprocess.py on a dataset that
    includes an "Other" gender label.
    """
    try:
        return int(gender_encoder.transform([gender])[0])
    except ValueError:
        return int(gender_encoder.transform(["Female"])[0])


def encode_bp(blood_pressure):
    try:
        return int(bp_encoder.transform([blood_pressure])[0])
    except ValueError:
        return int(bp_encoder.transform(["Normal"])[0])


def encode_cholesterol(cholesterol):
    try:
        return int(chol_encoder.transform([cholesterol])[0])
    except ValueError:
        return int(chol_encoder.transform(["Normal"])[0])


# ==========================================
# Database
# ==========================================





# ==========================================
# Save Prediction
# ==========================================

def save_prediction(
    user_id,
    age,
    gender,
    weight,
    symptoms,
    disease,
    confidence,
    severity,
    doctor
):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO history
        (
            user_id,
            age,
            gender,
            weight,
            symptoms,
            disease,
            confidence,
            severity,
            doctor
        )

        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,

        (
            user_id,
            age,
            gender,
            weight,
            symptoms,
            disease,
            confidence,
            severity,
            doctor
        )

    )

    conn.commit()
    cursor.close()
    conn.close()


# ==========================================
# Home
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================
# Register
# ==========================================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        fullname = request.form["fullname"].strip()
        email = request.form["email"].strip().lower()
        password = request.form["password"]

        hashed_password = generate_password_hash(password)

        age = request.form["age"]
        gender = request.form["gender"]
        weight = request.form["weight"]
        blood_group = request.form["blood_group"]

        conn = get_connection()
        cursor = conn.cursor()

        # Check if email already exists
        cursor.execute(
            "SELECT id FROM users WHERE email=%s",
            (email,)
        )

        user = cursor.fetchone()

        if user:
            cursor.close()
            conn.close()

            flash("Email is already registered.", "danger")
            return redirect("/register")

        # Save data in session
        session["register_data"] = {
            "fullname": fullname,
            "email": email,
            "password": hashed_password,
            "age": age,
            "gender": gender,
            "weight": weight,
            "blood_group": blood_group
        }

        # Generate OTP
        otp = str(random.randint(100000, 999999))

        session["register_otp"] = otp
        session["register_otp_time"] = time.time()

        # Send Email
        msg = Message(
            subject="Verify Your Email",
            sender=RESEND_FROM_EMAIL,
            recipients=[email]
        )

        msg.body = f"""
Hello {fullname},

Welcome to Smart Healthcare Assistant.

Your Email Verification OTP is:

{otp}

This OTP is valid for 5 minutes.

Do not share this OTP with anyone.

Thank You.
"""

        if not safe_send_mail(msg):
            cursor.close()
            conn.close()
            flash("Could not send verification email right now. Please try again.", "danger")
            return redirect("/register")

        cursor.close()
        conn.close()

        flash("OTP has been sent to your email.", "success")

        return redirect("/verify-register-otp")

    return render_template("register.html")
@app.route("/verify-register-otp", methods=["GET", "POST"])
def verify_register_otp():

    if "register_data" not in session:
        flash("Registration session expired.", "danger")
        return redirect("/register")

    if request.method == "POST":

        otp = request.form["otp"]

        # OTP Expiry (5 minutes)
        if time.time() - session.get("register_otp_time", 0) > 300:

            session.pop("register_data", None)
            session.pop("register_otp", None)
            session.pop("register_otp_time", None)

            flash("OTP Expired. Please register again.", "danger")
            return redirect("/register")

        # OTP Check
        if otp != session.get("register_otp"):

            flash("Invalid OTP.", "danger")
            return redirect("/verify-register-otp")

        data = session["register_data"]

        conn = get_connection()
        cursor = conn.cursor()

        try:

            cursor.execute("""
                INSERT INTO users
                (fullname,email,password,age,gender,weight,blood_group)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                data["fullname"],
                data["email"],
                data["password"],
                data["age"],
                data["gender"],
                data["weight"],
                data["blood_group"]
            ))

            conn.commit()

            session.pop("register_data", None)
            session.pop("register_otp", None)
            session.pop("register_otp_time", None)

            flash("Registration Successful. Please Login.", "success")

            return redirect("/login")

        except Exception as e:

            conn.rollback()

            print(e)

            flash("Registration Failed.", "danger")

        finally:

            cursor.close()
            conn.close()

    return render_template("verify_register_otp.html")

#login route
@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form["email"].strip()
        password = request.form["password"]

        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
         "SELECT * FROM users WHERE email=%s",
           (email,)
        )

        user = cursor.fetchone()

        cursor.close()
        conn.close()

        if not user:
            flash("No account found with this email.", "danger")
            return redirect("/login")

        if not check_password_hash(user["password"], password):
            flash("Enter correct password.", "danger")
            return redirect("/login")

        session["user_id"] = user["id"]
        session["fullname"] = user["fullname"]

        return redirect("/dashboard")

    return render_template("login.html")
@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():

    if request.method == "POST":

        email = request.form["email"].strip()

        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM users WHERE email=%s",
            (email,)
        )

        user = cursor.fetchone()

        cursor.close()
        conn.close()

        if not user:
            flash("Email not found.", "danger")
            return redirect("/forgot-password")

        otp = str(random.randint(100000, 999999))

        session["otp"] = otp
        session["reset_email"] = email
        session["otp_time"] = time.time()

        msg = Message(
            subject="Password Reset OTP",
            sender=RESEND_FROM_EMAIL,
            recipients=[email]
        )

        msg.body = f"""
Hello,

Your OTP for password reset is:

{otp}

Do not share this OTP with anyone.

Smart Healthcare Assistant
"""

        if not safe_send_mail(msg):
            flash("Could not send OTP email right now. Please try again.", "danger")
            return redirect("/forgot-password")

        flash("OTP sent successfully.", "success")

        return redirect("/verify-otp")

    return render_template("forgot_password.html")
@app.route("/verify-otp", methods=["GET", "POST"])
def verify_otp():

    if request.method == "POST":

        otp = request.form["otp"]
        if time.time() - session.get("otp_time", 0) > 300:

            flash("OTP Expired. Please request a new OTP.", "danger")

            session.pop("otp", None)
            session.pop("otp_time", None)

            return redirect("/forgot-password")

        if otp == session.get("otp"):

            return redirect("/reset-password")

        flash("Invalid OTP", "danger")

    return render_template("verify_otp.html")
@app.route("/reset-password", methods=["GET", "POST"])
def reset_password():

    if request.method == "POST":

        password = generate_password_hash(
         request.form["password"]
       )

        email = session.get("reset_email")

        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE users
            SET password=%s
            WHERE email=%s
            """,
            (password, email)
        )

        conn.commit()

        cursor.close()
        conn.close()

        session.pop("otp", None)
        session.pop("reset_email", None)
        session.pop("otp_time", None)

        flash("Password Updated Successfully", "success")

        return redirect("/login")

    return render_template("reset_password.html")
# ==========================================
# Logout
# ==========================================

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


# ==========================================
# Dashboard
# ==========================================

@app.route("/dashboard")
def dashboard():
    
    
    if "user_id" not in session:
        return redirect("/login")

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
       SELECT *
       FROM history
      WHERE user_id=%s
      ORDER BY prediction_date DESC
     """, (session["user_id"],))

    history = cursor.fetchall()

    cursor.close()
    conn.close()

    return render_template(
        "dashboard.html",
        fullname=session["fullname"],
        history=history
    )


# ==========================================
# Predict Page
# ==========================================

@app.route("/predict")
def predict():

    if "user_id" not in session:
        return redirect("/login")

    symptoms = []

    for feature in feature_names:
        if feature not in NON_SYMPTOM_FEATURES:
            symptoms.append(feature.title())

    symptoms.sort()

    return render_template(
        "predict.html",
        symptoms=symptoms
    )


# ==========================================
# Result Page
# ==========================================

@app.route("/result", methods=["POST"])
def result():

    if "user_id" not in session:
        return redirect("/login")
        

    # -----------------------------
    # Get Form Data
    # -----------------------------
    age = int(request.form["age"])
    gender = request.form["gender"]
    bmi = float(request.form["bmi"])
    blood_pressure = request.form["blood_pressure"]
    cholesterol = request.form["cholesterol"]
    weight = float(request.form["weight"])
   
    if bmi < 18.5:
      bmi_status = "Underweight"
    elif bmi < 25:
     bmi_status = "Normal"

    elif bmi < 30:
      bmi_status = "Overweight"

    else:
      bmi_status = "Obese"
    # predict.html submits up to 5 individual symptom dropdowns
    # (symptom1 .. symptom5) instead of a multi-select list.
    symptoms = [
        request.form.get(f"symptom{i}", "").strip()
        for i in range(1, 6)
    ]
    symptoms = [s for s in symptoms if s]  # drop empty selections

    if not symptoms:
        flash("Please select at least one symptom.")
        return redirect("/predict")

    # -----------------------------
    # Create Input Dictionary
    # -----------------------------
    input_data = {feature: 0 for feature in feature_names}

    input_data["Age"] = age
    input_data["BMI"] = bmi
    input_data["Gender"] = encode_gender(gender)
    input_data["Blood_Pressure"] = encode_bp(blood_pressure)
    input_data["Cholesterol_Level"] = encode_cholesterol(cholesterol)

    for symptom in symptoms:
        symptom = symptom.lower().strip()
        if symptom in input_data:
            input_data[symptom] = 1

    # -----------------------------
    # Convert To DataFrame (column order MUST match training)
    # -----------------------------
    X = pd.DataFrame([input_data], columns=feature_names)

    # -----------------------------
    # Random Forest + XGBoost Predictions (2-model ensemble)
    # -----------------------------
   # -----------------------------
# Random Forest Prediction
# rf_probs = rf_model.predict_proba(X)[0------------------
# Stacking Model Prediction
    final_probs = stack_model.predict_proba(X)[0]

    prediction = int(np.argmax(final_probs))
    disease = disease_encoder.inverse_transform([prediction])[0]
    confidence = round(float(final_probs[prediction]) * 100, 2)

    # -----------------------------
    # Top 3 Predictions
    # -----------------------------
    top3_idx = np.argsort(final_probs)[::-1][:3]

    top3_predictions = []
    for i in top3_idx:
        top3_predictions.append({
            "disease": disease_encoder.inverse_transform([i])[0],
            "confidence": round(float(final_probs[i]) * 100, 2)
        })

    # -----------------------------
    # Knowledge Base
    # -----------------------------
    precautions = get_precautions(disease)
    medicines = get_medicines(disease)
    doctor = get_doctor(disease)
    severity = get_severity(disease)
    diet = get_diet(disease)
    workout = get_workout(disease)

    # -----------------------------
    # Save Prediction
    # -----------------------------
    symptom_string = ", ".join(symptoms)

    save_prediction(
        session["user_id"],
        age,
        gender,
        weight,
        symptom_string,
        disease,
        confidence,
        severity,
        doctor
    )

    # -----------------------------
    # Store Last Result
    # -----------------------------
    session["last_result"] = {
        "disease": disease,
        "confidence": confidence,
        "severity": severity,
        "doctor": doctor,
        "age": age,
        "gender": gender,
        "weight": weight,
        "bmi": bmi
    }

    # -----------------------------
    # Show Result
    # -----------------------------
    return render_template(
    "result.html",
    disease=disease,
    confidence=confidence,
    severity=severity,
    doctor=doctor,
    precautions=precautions,
    medicines=medicines,
    diet=diet,
    workout=workout,

    age=age,
    gender=gender,
    weight=weight,
    bmi=bmi
)


# ==========================================
# Download Report (PDF)
# ==========================================

@app.route("/download_report")
def download_report():

    if "user_id" not in session:
        return redirect("/login")

    last_result = session.get("last_result")

    if not last_result:
        flash("No recent prediction found. Please run a symptom check first.")
        return redirect("/predict")

    disease = last_result["disease"]
    confidence = last_result["confidence"]
    severity = last_result["severity"]
    doctor = last_result["doctor"]
    age = last_result.get("age")
    gender = last_result.get("gender")
    weight = last_result.get("weight")
    bmi = last_result.get("bmi")

    precautions = get_precautions(disease)
    diet = get_diet(disease)
    workout = get_workout(disease)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()
    elements = []

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        textColor=colors.HexColor("#2563eb")
    )

    generated_style = ParagraphStyle(
        "GeneratedOn",
        parent=styles["Normal"],
        textColor=colors.HexColor("#5b7186"),
        fontSize=9
    )

    elements.append(Paragraph("Healthcare AI", title_style))
    elements.append(Paragraph("AI Disease Prediction Report", styles["Heading3"]))
    elements.append(Paragraph(
        f"Generated on: {datetime.now().strftime('%d %b %Y, %I:%M %p')}",
        generated_style
    ))
    elements.append(Spacer(1, 16))

    elements.append(Paragraph(f"<b>Predicted Disease:</b> {disease}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Confidence:</b> {confidence}%", styles["Normal"]))
    elements.append(Paragraph(f"<b>Risk Level:</b> {severity}", styles["Normal"]))
    elements.append(Spacer(1, 14))

    elements.append(Paragraph("Patient Information", styles["Heading2"]))

    patient_data = [
        ["Age", str(age) if age is not None else "-"],
        ["Gender", gender or "-"],
        ["Weight", f"{weight} kg" if weight is not None else "-"],
        ["BMI", str(bmi) if bmi is not None else "-"],
    ]

    table = Table(patient_data, colWidths=[150, 300])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef4fb")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d0d7e2")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 16))

    def bullet_section(title, items):
        elements.append(Paragraph(title, styles["Heading2"]))
        if items:
            for item in items:
                elements.append(Paragraph(f"&bull; {item}", styles["Normal"]))
        else:
            elements.append(Paragraph("No recommendations available.", styles["Normal"]))
        elements.append(Spacer(1, 12))

    bullet_section("Diet Plan", diet)
    bullet_section("Workout Plan", workout)
    bullet_section("Precautions", precautions)

    elements.append(Paragraph(f"<b>Recommended Doctor:</b> {doctor}", styles["Normal"]))

    doc.build(elements)
    buffer.seek(0)

    safe_disease = "".join(c for c in disease if c.isalnum() or c in (" ", "_")).strip().replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"health_report_{safe_disease}_{timestamp}.pdf",
        mimetype="application/pdf"
    )


# ==========================================
# History
# ==========================================

@app.route("/history")
def history():

    if "user_id" not in session:
        return redirect("/login")

    user_id = session["user_id"]

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM history
        WHERE user_id=%s
        ORDER BY prediction_date DESC
    """, (user_id,))

    history_rows = cursor.fetchall()

    cursor.close()
    conn.close()

    return render_template(
        "history.html",
        history=history_rows
    )
# ==========================================
# Profile
# ==========================================

@app.route("/profile")
def profile():

    if "user_id" not in session:
        return redirect("/login")

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
      SELECT * FROM users
      WHERE id=%s
    """, (session["user_id"],))

    user = cursor.fetchone()

    cursor.close()
    conn.close()

    

    return render_template("profile.html", user=user)



# ============================
# Google Login
# ============================
@app.route("/google-login")
def google_login():
    redirect_uri = url_for("authorize", _external=True)
    return google.authorize_redirect(
        redirect_uri,
        prompt="select_account"
    )


# ============================
# Google Callback
# ============================
@app.route("/login/google/authorized")
def authorize():

    try:
        token = google.authorize_access_token()

        user = token.get("userinfo")

        if not user:
            return "Unable to fetch Google user information."

        email = user["email"]
        name = user["name"]

        conn = get_connection()
        cursor = conn.cursor()

        # Check if user already exists
        cursor.execute(
            """
            SELECT *
            FROM users
            WHERE email = %s
            """,
            (email,)
        )

        existing = cursor.fetchone()

        # If user does not exist, create new account
        if existing is None:

            cursor.execute(
                """
                INSERT INTO users (fullname, email, password)
                VALUES (%s, %s, %s)
                RETURNING id, fullname, email
                """,
                (
                    name,
                    email,
                    "google_login"
                )
            )

            conn.commit()

            existing = cursor.fetchone()

        cursor.close()
        conn.close()

        session["user_id"] = existing["id"]
        session["fullname"] = existing["fullname"]
        session["email"] = existing["email"]

        return redirect(url_for("dashboard"))

    except Exception as e:
        return f"Google Login Error: {str(e)}"

    try:
        token = google.authorize_access_token()

        user = token.get("userinfo")

        if not user:
            return "Unable to fetch user info"

        email = user["email"]
        name = user["name"]

        conn = get_connection()

        existing = conn.execute(
            "SELECT * FROM users WHERE email=%s",
            (email,)
        ).fetchone()

        if existing is None:

            conn.execute("""
                INSERT INTO users(fullname,email,password)
                VALUES(%s,%s,%s)
            """, (
                name,
                email,
                "google_login"
            ))

            conn.commit()

            existing = conn.execute(
                "SELECT * FROM users WHERE email=%s",
                (email,)
            ).fetchone()

        conn.close()

        session["user_id"] = existing["id"]
        session["fullname"] = existing["fullname"]
        session["email"] = existing["email"]

        return redirect(url_for("dashboard"))

    except Exception as e:
        return str(e)
@app.route("/test-mail")
def test_mail():
    msg = Message(
        subject="Smart Healthcare Assistant",
        sender=RESEND_FROM_EMAIL,
        recipients=["healthcare0ai@gmail.com"]
    )

    msg.body = """
Hello,

Congratulations 🎉

Your Resend email configuration is working successfully.

Smart Healthcare Assistant
"""

    if safe_send_mail(msg):
        return "Email Sent Successfully"

    return "Email failed to send. Check Railway logs.", 500
# ==========================================
# Run App
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)