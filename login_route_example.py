from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.security import check_password_hash

app = Flask(__name__)
app.secret_key = "your-secret-key-here"  # required for flash() to work

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        # --- Replace this with your actual DB lookup ---
        user = get_user_by_email(email)

        if not user:
            flash("No account found with this email.", "error")
            return redirect(url_for("login"))

        if not check_password_hash(user.password_hash, password):
            flash("Enter correct password.", "error")
            return redirect(url_for("login"))

        # Success — log the user in
        # login_user(user)  # if using Flask-Login
        flash("Logged in successfully.", "success")
        return redirect(url_for("dashboard"))

    return render_template("login.html")
