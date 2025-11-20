from flask import Flask, render_template, request, redirect, url_for, session
import sqlite3
import pickle
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # for sessions; fine for project/demo

# ====== DNA COLUMN CONFIG (must match MainFile.py) ======
DNA_COLUMNS = [
    'Seq_Window_1', 'Seq_Window_2', 'Promoter_Seq', 'Exon_1_Seq',
    'Mutation_Site_Seq', 'Enhancer_Seq', 'Intron_1_Seq', 'CpG_Island_Seq',
    'Repeat_Seq', 'Motif_1', 'Motif_2', 'DNA_Segment_1',
    'DNA_Segment_2', 'DNA_Segment_3', 'DNA_Segment_4'
]

DINUCLEOTIDES = [a + b for a in "ACGT" for b in "ACGT"]

# ====== DB INIT ======
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL
                )''')

    # Predictions table (new)
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT,
                    disease TEXT,
                    confidence REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )''')
    conn.commit()
    conn.close()

init_db()

# ====== DISEASE RECOMMENDATIONS ======
recommendations = {
    'Breast Cancer': [
        "Schedule regular mammograms and clinical breast exams.",
        "Discuss genetic testing (e.g., BRCA1/2) with your doctor if there's a family history.",
        "Maintain a healthy weight and reduce alcohol consumption.",
        "Perform monthly breast self-exams and report any changes."
    ],
    'Heart Failure': [
        "Limit sodium intake to reduce fluid buildup.",
        "Take prescribed medications regularly and monitor symptoms.",
        "Track weight daily to detect fluid retention early.",
        "Engage in moderate physical activity as advised by your cardiologist."
    ],
    'Diabetes': [
        "Monitor your blood sugar levels regularly.",
        "Maintain a balanced diet low in refined carbs and sugars.",
        "Exercise regularly to improve insulin sensitivity.",
        "Consult your doctor about medication or insulin therapy."
    ],
    'Alzheimer': [
        "Establish a routine and safe environment to reduce confusion.",
        "Stay mentally active with puzzles, reading, or memory exercises.",
        "Engage in regular physical and social activities.",
        "Consult a neurologist for medication options to slow progression."
    ],
    'Lung Cancer': [
        "Stop smoking immediately and avoid secondhand smoke.",
        "Get imaging tests (CT scans or X-rays) as advised by your doctor.",
        "Discuss treatment options like surgery, chemo, or immunotherapy.",
        "Maintain good nutrition and rest to support overall health."
    ],
    'Parkinson': [
        "Work with a neurologist to manage medications and symptoms.",
        "Join a physical therapy program to maintain mobility and balance.",
        "Eat a high-fiber diet and stay hydrated to prevent constipation.",
        "Consider support groups for emotional and mental well-being."
    ],
    'Stroke': [
        "Seek immediate medical attention for any stroke symptoms.",
        "Take blood thinners or other medications as prescribed.",
        "Participate in rehabilitation therapies to regain function.",
        "Manage risk factors like blood pressure, cholesterol, and diabetes."
    ],
    'Prostate Cancer': [
        "Discuss PSA screening and biopsy options with your doctor.",
        "Consider active surveillance if cancer is low risk.",
        "Explore treatment options including surgery, radiation, or hormone therapy.",
        "Maintain a healthy diet rich in fruits, vegetables, and low-fat foods."
    ],
    'Asthma': [
        "Avoid triggers such as pollen, dust, and smoke.",
        "Use inhalers and medications exactly as prescribed.",
        "Monitor your breathing and keep a symptom diary.",
        "Develop an asthma action plan with your healthcare provider."
    ],
    'Osteoporosis': [
        "Ensure adequate intake of calcium and vitamin D.",
        "Engage in weight-bearing and strength-training exercises.",
        "Avoid smoking and limit alcohol consumption.",
        "Get bone density scans as recommended by your doctor."
    ]
}

default_recs = [
    "Consult a qualified doctor for a detailed medical evaluation.",
    "Maintain a balanced diet and regular physical activity.",
    "Avoid smoking and limit alcohol consumption.",
    "Go for regular health checkups and screenings as advised.",
]

# High-risk diseases (for the “High Risk Cases” card)
HIGH_RISK_DISEASES = [
    "Heart Failure", "Stroke", "Lung Cancer",
    "Breast Cancer", "Prostate Cancer"
]

# ====== LOAD MODEL / SCALER / LABEL ENCODER / FEATURE COLUMNS ======
with open('dna_model.pkl', 'rb') as f:
    dna_model = pickle.load(f)

with open('dna_scaler.pkl', 'rb') as f:
    dna_scaler = pickle.load(f)

with open('dna_label_encoder.pkl', 'rb') as f:
    dna_label_encoder = pickle.load(f)

with open('dna_feature_columns.pkl', 'rb') as f:
    dna_feature_columns = pickle.load(f)

# ====== FEATURIZATION (same logic as in MainFile.py) ======
def featurize_sequence(seq: str) -> dict:
    if seq is None:
        seq = ""
    seq = seq.strip().upper()
    seq = "".join([c for c in seq if c in "ACGT"])

    length = len(seq)
    if length == 0:
        length = 1
        seq_clean = ""
    else:
        seq_clean = seq

    counts = {base: seq_clean.count(base) for base in "ACGT"}
    mono_props = {f"prop_{b}": counts[b] / length for b in "ACGT"}
    gc_content = (counts["G"] + counts["C"]) / length

    d_counts = {d: 0 for d in DINUCLEOTIDES}
    for i in range(len(seq_clean) - 1):
        pair = seq_clean[i:i+2]
        if pair in d_counts:
            d_counts[pair] += 1

    total_pairs = max(len(seq_clean) - 1, 1)
    d_freqs = {f"di_{d}": d_counts[d] / total_pairs for d in DINUCLEOTIDES}

    features = {
        "length": float(length),
        "gc_content": gc_content,
    }
    features.update(mono_props)
    features.update(d_freqs)
    return features


def featurize_input_sequences(seq_dict: dict) -> np.ndarray:
    """
    seq_dict: {original_column_name: sequence_string}
    Returns a (1, n_features) numpy array ordered according to dna_feature_columns.
    """
    row_features = {}
    for col in DNA_COLUMNS:
        seq_val = seq_dict.get(col, "")
        seq_feats = featurize_sequence(seq_val)
        for k, v in seq_feats.items():
            row_features[f"{col}_{k}"] = v

    feature_vector = []
    for col_name in dna_feature_columns:
        feature_vector.append(float(row_features.get(col_name, 0.0)))

    return np.array(feature_vector, dtype=float).reshape(1, -1)

# ====== DASHBOARD STATS HELPERS ======

def get_dashboard_stats():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()

    # Total analyses
    c.execute("SELECT COUNT(*) FROM predictions")
    total_analyses = c.fetchone()[0] or 0

    # High-risk cases
    placeholders = ",".join("?" for _ in HIGH_RISK_DISEASES)
    c.execute(
        f"SELECT COUNT(*) FROM predictions WHERE disease IN ({placeholders})",
        HIGH_RISK_DISEASES
    )
    high_risk_cases = c.fetchone()[0] or 0

    # Average confidence (0–1) → convert to %
    c.execute("SELECT AVG(confidence) FROM predictions")
    avg_conf = c.fetchone()[0]
    if avg_conf is None:
        accuracy_rate = 0.0
    else:
        accuracy_rate = float(avg_conf) * 100.0

    # Pending reviews: for demo, treat high-risk as "needs review"
    pending_reviews = high_risk_cases

    conn.close()
    return total_analyses, accuracy_rate, high_risk_cases, pending_reviews


def get_recent_analyses(limit=5):
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute(
        "SELECT disease, confidence, created_at FROM predictions "
        "ORDER BY created_at DESC LIMIT ?",
        (limit,)
    )
    rows = c.fetchall()
    conn.close()
    return rows

# ====== AUTH ROUTES ======

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])

        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return render_template('register.html', error="Username already exists.")
        finally:
            conn.close()

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password_input = request.form['password']

        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT password FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        conn.close()

        if user and check_password_hash(user[0], password_input):
            session['username'] = username
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error="Invalid username or password.")

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

# ====== MAIN PREDICTION ROUTE ======

@app.route('/', methods=['GET', 'POST'])
def home():
    if 'username' not in session:
        return redirect(url_for('login'))

    prediction = None
    recs = None
    error = None

    if request.method == 'POST':
        try:
            # collect sequences
            seq_dict = {}
            for feature in DNA_COLUMNS:
                seq_dict[feature] = request.form.get(feature, "")

            # featurize + scale
            feature_vec = featurize_input_sequences(seq_dict)
            feature_scaled = dna_scaler.transform(feature_vec)

            # predict class index
            pred_idx = dna_model.predict(feature_scaled)[0]
            prediction = dna_label_encoder.inverse_transform([pred_idx])[0]

            # confidence (max probability)
            max_conf = None
            if hasattr(dna_model, "predict_proba"):
                proba_vec = dna_model.predict_proba(feature_scaled)[0]
                max_conf = float(np.max(proba_vec))
            else:
                max_conf = 1.0  # fallback

            # save prediction in DB
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute(
                "INSERT INTO predictions (username, disease, confidence) VALUES (?, ?, ?)",
                (session.get('username'), prediction, max_conf)
            )
            conn.commit()
            conn.close()

            # recommendations
            recs = recommendations.get(prediction, default_recs)

        except Exception as e:
            error = str(e)

    # Dashboard stats & history
    total_analyses, accuracy_rate, high_risk_cases, pending_reviews = get_dashboard_stats()
    recent_analyses = get_recent_analyses(limit=5)

    return render_template(
        'index.html',
        feature_names=DNA_COLUMNS,
        username=session.get('username'),
        prediction=prediction,
        recommendations=recs,
        error=error,
        total_analyses=total_analyses,
        accuracy_rate=accuracy_rate,
        high_risk_cases=high_risk_cases,
        pending_reviews=pending_reviews,
        recent_analyses=recent_analyses,
    )

if __name__ == '__main__':
    app.run(debug=True)
