# app.py
from flask import Flask, render_template, request, Response, jsonify, abort, url_for
import joblib
import numpy as np
import sqlite3
from datetime import datetime
import csv
import io
import os
import secrets
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from flask_socketio import SocketIO

# Ensure vader lexicon available
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except Exception:
    nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get("FLASK_SECRET", "secret!")
socketio = SocketIO(app, cors_allowed_origins="*")

# Admin token for revealing identities (override with env var in prod)
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "HR2025")

# ===== Load numeric model =====
MODEL_PATH = "model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("model.pkl not found!")

model = joblib.load(MODEL_PATH)

# ===== DB Setup =====
DB_PATH = "employees.db"

def get_db_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_conn()
    c = conn.cursor()

    # identity: stores PII (name, emp_id) linked to anon_code
    c.execute('''CREATE TABLE IF NOT EXISTS identity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    anon_code TEXT UNIQUE,
                    name TEXT,
                    emp_id TEXT,
                    created_at TEXT
                )''')

    # wellness: survey and result (does NOT duplicate PII when hide_identity=1)
    c.execute('''CREATE TABLE IF NOT EXISTS wellness (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT,
                    anon_code TEXT,
                    age INTEGER,
                    gender TEXT,
                    marital_status INTEGER,
                    work_stress INTEGER,
                    counselling_support INTEGER,
                    confidentiality_assurance INTEGER,
                    job_satisfaction INTEGER,
                    coworkers INTEGER,
                    supervisor INTEGER,
                    leave_policy INTEGER,
                    promotion INTEGER DEFAULT 0,
                    sentiment REAL,
                    hide_identity INTEGER DEFAULT 0,
                    risk TEXT,
                    confidence REAL,
                    suggestion TEXT
                )''')

    # audit log for identity requests from HR
    c.execute('''CREATE TABLE IF NOT EXISTS identity_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    anon_code TEXT,
                    requested_by TEXT,
                    reason TEXT,
                    requested_at TEXT
                )''')

    conn.commit()
    conn.close()

init_db()

# ===== Utilities =====
def generate_suggestions(pred_label, inputs):
    sugg = []
    if pred_label == 1:
        sugg += ["Take a short break", "Talk to HR", "Consider counselling", "Spend time with family"]
    else:
        sugg += ["Maintain healthy routine", "Take regular breaks", "Stay socially active"]

    if inputs.get("work_stress", 0) >= 2:
        sugg.append("Try stress relaxation activities")
    if inputs.get("counselling_support", 1) == 0:
        sugg.append("Ask HR about counselling support")
    if inputs.get("confidentiality_assurance", 1) == 0:
        sugg.append("Raise concerns anonymously if needed")
    if inputs.get("leave_policy", 1) == 0:
        sugg.append("Discuss flexible leave with HR")

    # remove duplicates while preserving order
    return list(dict.fromkeys(sugg))

def generate_anon_code():
    return "ANON-" + secrets.token_hex(3)

# ===== Routes =====
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read inputs (form format unchanged)
        name = request.form.get("name", "").strip()
        emp_id = request.form.get("emp_id", "").strip()
        age = int(request.form["age"])
        gender = int(request.form["gender"])
        marital_status = int(request.form["marital_status"])
        work_stress = int(request.form["work_stress"])
        counselling_support = int(request.form["counselling_support"])
        confidentiality_assurance = int(request.form["confidentiality_assurance"])
        job_satisfaction = int(request.form["job_satisfaction"])
        coworkers = int(request.form["coworkers"])
        supervisor = int(request.form["supervisor"])
        leave_policy = int(request.form["leave_policy"])
        promotion = int(request.form.get("promotion", 0))
        hide_identity = 1 if request.form.get("private_submission") == "1" else 0
        mood_text = request.form.get("mood_text", "").strip()

        # Validate age
        if age < 21 or age > 60:
            return "Bad Request: Age must be between 21 and 60", 400

        # Sentiment (optional)
        sentiment_compound = None
        if mood_text:
            vs = sia.polarity_scores(mood_text)
            sentiment_compound = float(vs["compound"])

        # Model input (preserve order)
        X = np.array([[age, gender, marital_status, work_stress,
                       counselling_support, confidentiality_assurance,
                       job_satisfaction, coworkers, supervisor, leave_policy]])

        # Model prediction / probability
        pred_label = int(model.predict(X)[0])
        rf_proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            # proba[1] is probability of positive class (risk)
            rf_proba = float(proba[1]) if len(proba) > 1 else float(proba[0])
        else:
            rf_proba = 1.0 if pred_label == 1 else 0.0

        # Combine with text sentiment
        if sentiment_compound is not None:
            text_score = (sentiment_compound + 1.0) / 2.0
        else:
            text_score = 0.5

        combined_score = 0.75 * rf_proba + 0.25 * text_score
        combined_confidence_pct = round(combined_score * 100.0, 2)
        final_label = 1 if combined_score >= 0.5 else 0

        if final_label == 1:
            risk_text = "⚠ High Risk"
        else:
            if 40 <= combined_confidence_pct < 70:
                risk_text = "🟠 Medium Risk"
            else:
                risk_text = "✅ Low Risk"

        inputs_map = {
            "work_stress": work_stress,
            "counselling_support": counselling_support,
            "confidentiality_assurance": confidentiality_assurance,
            "leave_policy": leave_policy
        }
        suggestions = generate_suggestions(final_label, inputs_map)

        # Save identity (PII) in identity table with anon_code (always saved so admin can map later)
        anon_code = generate_anon_code()
        conn = get_db_conn()
        c = conn.cursor()
        c.execute("""INSERT INTO identity (anon_code, name, emp_id, created_at)
                     VALUES (?,?,?,?)""",
                  (anon_code, name, emp_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

        # Save wellness row (no PII fields here; link via anon_code)
        c.execute("""INSERT INTO wellness
                  (created_at, anon_code, age, gender, marital_status, work_stress,
                   counselling_support, confidentiality_assurance, job_satisfaction,
                   coworkers, supervisor, leave_policy, promotion, sentiment,
                   hide_identity, risk, confidence, suggestion)
                  VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                  (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), anon_code, age,
                   "Male" if gender == 1 else "Female",
                   marital_status, work_stress, counselling_support,
                   confidentiality_assurance, job_satisfaction, coworkers,
                   supervisor, leave_policy, promotion,
                   sentiment_compound if sentiment_compound is not None else None,
                   hide_identity, risk_text, combined_confidence_pct,
                   " | ".join(suggestions)))
        conn.commit()
        wellness_id = c.lastrowid
        conn.close()

        # Emit real-time event to dashboards (do not include PII if hidden)
        payload = {
            "id": wellness_id,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "anon_code": anon_code,
            "age": age,
            "gender": "Male" if gender == 1 else "Female",
            "promotion": promotion,
            "risk": risk_text,
            "confidence": combined_confidence_pct,
            "suggestion": " | ".join(suggestions)
        }
        if hide_identity == 0:
            payload["name"] = name
            payload["emp_id"] = emp_id
        else:
            payload["name"] = "Private Submission"
            payload["emp_id"] = ""

        socketio.emit("new_record", payload)

        # Render result page for user (displaying "Private Submission" if they hid their identity)
        display_name = "Private Submission" if hide_identity == 1 else name
        display_emp = "Private Submission" if hide_identity == 1 else emp_id

        return render_template("result.html",
                               name=display_name, emp_id=display_emp,
                               result=risk_text,
                               confidence=combined_confidence_pct,
                               suggestions=suggestions,
                               anon_code=anon_code)
    except Exception as e:
        return f"Bad Request: {str(e)}", 400

# Dashboard: show only non-private wellness rows (HR view)
@app.route("/dashboard", methods=["GET"])
def dashboard():
    conn = get_db_conn()
    c = conn.cursor()
    c.execute("SELECT * FROM wellness ORDER BY id DESC")
    rows_all = c.fetchall()
    conn.close()

    # For HR dashboard we will show only rows but mark hidden ones (we'll still provide counts including hidden)
    # Here we pass all rows to the template but template will show name/emp as anon if hide_identity=1
    # Compute summary metrics
    high_count = sum(1 for r in rows_all if r["risk"] and "High Risk" in r["risk"])
    medium_count = sum(1 for r in rows_all if r["risk"] and "Medium Risk" in r["risk"])
    low_count = sum(1 for r in rows_all if r["risk"] and "Low Risk" in r["risk"])

    male_count = sum(1 for r in rows_all if r["gender"] == "Male")
    female_count = sum(1 for r in rows_all if r["gender"] == "Female")
    total_count = len(rows_all)

    return render_template("dashboard.html",
                           rows=rows_all,
                           high_count=high_count,
                           medium_count=medium_count,
                           low_count=low_count,
                           male_count=male_count,
                           female_count=female_count,
                           total_count=total_count)

# Male dashboard (format unchanged)
@app.route("/male_dashboard", methods=["GET"])
def male_dashboard():
    conn = get_db_conn()
    c = conn.cursor()
    c.execute("SELECT w.*, i.anon_code, i.name as real_name, i.emp_id as real_emp_id FROM wellness w LEFT JOIN identity i ON w.anon_code=i.anon_code WHERE w.gender='Male' ORDER BY w.id DESC")
    rows = c.fetchall()
    conn.close()

    high = sum(1 for r in rows if r["risk"] and "High Risk" in r["risk"])
    medium = sum(1 for r in rows if r["risk"] and "Medium Risk" in r["risk"])
    low = sum(1 for r in rows if r["risk"] and "Low Risk" in r["risk"])

    counselling = None
    if high > 0:
        counselling = "⚠ Male employees in high stress – recommend counselling sessions, stress workshops, HR check-ins."

    # Template expects r['name'] and r['emp_id'] for display; if hide_identity==1 we show anon (template uses the values passed)
    # We'll attach display_name & display_emp in template using anon_code when hidden.
    return render_template("male_dashboard.html", rows=rows, high=high, medium=medium, low=low, counselling=counselling)

# Female dashboard (format unchanged)
@app.route("/female_dashboard", methods=["GET"])
def female_dashboard():
    conn = get_db_conn()
    c = conn.cursor()
    c.execute("SELECT w.*, i.anon_code, i.name as real_name, i.emp_id as real_emp_id FROM wellness w LEFT JOIN identity i ON w.anon_code=i.anon_code WHERE w.gender='Female' ORDER BY w.id DESC")
    rows = c.fetchall()
    conn.close()

    high = sum(1 for r in rows if r["risk"] and "High Risk" in r["risk"])
    medium = sum(1 for r in rows if r["risk"] and "Medium Risk" in r["risk"])
    low = sum(1 for r in rows if r["risk"] and "Low Risk" in r["risk"])

    counselling = None
    if high > 0:
        counselling = "⚠ Female employees in high stress – suggest health counselling, flexible work hours, and wellness programs."

    return render_template("female_dashboard.html", rows=rows, high=high, medium=medium, low=low, counselling=counselling)

# Employee trend by anon_code
@app.route("/employee/<emp_id>", methods=["GET"])
def employee_trend(emp_id):
    conn = get_db_conn()
    c = conn.cursor()
    c.execute("SELECT created_at, confidence, risk, anon_code FROM wellness WHERE anon_code=? ORDER BY created_at ASC", (emp_id,))
    rows = c.fetchall()
    conn.close()
    return render_template("employee_trend.html", rows=rows, emp_id=emp_id)

# Export CSV - if admin token provided include real name/emp_id in file, otherwise include anon_code only
@app.route("/export", methods=["GET"])
def export_csv():
    token = request.args.get("token", None)
    conn = get_db_conn()
    c = conn.cursor()
    if token == ADMIN_TOKEN:
        c.execute("SELECT w.*, i.name as real_name, i.emp_id as real_emp_id FROM wellness w LEFT JOIN identity i ON w.anon_code=i.anon_code ORDER BY w.id DESC")
    else:
        c.execute("SELECT w.*, i.anon_code FROM wellness w LEFT JOIN identity i ON w.anon_code=i.anon_code ORDER BY w.id DESC")
    rows = c.fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)
    header = ["id","created_at","anon_code_or_name","emp_id","age","gender","marital_status",
              "work_stress","counselling_support","confidentiality_assurance",
              "job_satisfaction","coworkers","supervisor","leave_policy","promotion",
              "sentiment","hide_identity","risk","confidence","suggestion"]
    writer.writerow(header)
    for r in rows:
        if token == ADMIN_TOKEN:
            name_field = r["real_name"] if "real_name" in r.keys() and r["real_name"] else r["anon_code"]
            emp_field = r["real_emp_id"] if "real_emp_id" in r.keys() and r["real_emp_id"] else ""
        else:
            name_field = r["anon_code"]
            emp_field = ""
        writer.writerow([r["id"], r["created_at"], name_field, emp_field, r["age"], r["gender"],
                         r["marital_status"], r["work_stress"], r["counselling_support"], r["confidentiality_assurance"],
                         r["job_satisfaction"], r["coworkers"], r["supervisor"], r["leave_policy"], r["promotion"],
                         r["sentiment"], r["hide_identity"], r["risk"], r["confidence"], r["suggestion"]])

    csv_data = output.getvalue()
    output.close()
    return Response(csv_data, mimetype="text/csv",
                    headers={"Content-Disposition": "attachment;filename=employee_wellness.csv"})

# Request identity endpoint - logs HR request into identity_requests table
@app.route("/request_identity", methods=["POST"])
def request_identity():
    data = request.get_json() or {}
    anon_code = data.get("anon_code")
    requester = data.get("requested_by", "HR")
    reason = data.get("reason", "")
    if not anon_code:
        return jsonify({"status": "error", "message": "anon_code required"}), 400

    conn = get_db_conn()
    c = conn.cursor()
    c.execute("INSERT INTO identity_requests (anon_code, requested_by, reason, requested_at) VALUES (?,?,?,?)",
              (anon_code, requester, reason, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()
    return jsonify({"status": "ok", "message": "Request logged. Admin will review."})

# Admin page - shows wellness rows (anon/real depending on hide_identity) and identity requests
@app.route("/admin", methods=["GET"])
def admin():
    conn = get_db_conn()
    c = conn.cursor()
    # Fetch wellness rows but do not expose PII here directly if hide_identity==1
    c.execute("""SELECT w.*, i.name as real_name, i.emp_id as real_emp_id, i.anon_code
                 FROM wellness w LEFT JOIN identity i ON w.anon_code=i.anon_code
                 ORDER BY w.id DESC""")
    rows = c.fetchall()
    # Fetch identity_requests for admin review panel
    c.execute("SELECT * FROM identity_requests ORDER BY id DESC")
    requests = c.fetchall()
    conn.close()
    # admin.html will display requests and rows; to actually reveal PII admin must use /admin/reveal?token=...&code=...
    return render_template("admin.html", rows=rows, requests=requests, admin_token=ADMIN_TOKEN)

# Admin reveal endpoint: returns PII for a given anon_code IF token matches
@app.route("/admin/reveal", methods=["GET"])
def admin_reveal():
    token = request.args.get("token")
    code = request.args.get("code")
    if token != ADMIN_TOKEN:
        return abort(401)
    if not code:
        return jsonify({"status": "error", "message": "code required"}), 400
    conn = get_db_conn()
    c = conn.cursor()
    c.execute("SELECT * FROM identity WHERE anon_code=?", (code,))
    row = c.fetchone()
    conn.close()
    if not row:
        return jsonify({"status": "error", "message": "anon_code not found"}), 404
    # Return real identity to admin (JSON)
    return jsonify({"anon_code": row["anon_code"], "name": row["name"], "emp_id": row["emp_id"], "created_at": row["created_at"]})

# Socket test endpoint (optional)
@app.route("/socket_test")
def socket_test():
    return "Socket server running."

if __name__ == "__main__":
    socketio.run(app, debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
