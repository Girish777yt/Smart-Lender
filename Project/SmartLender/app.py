from flask import Flask, render_template, request, redirect, url_for, session, flash
import joblib
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime

app = Flask(__name__)
# NOTE: Change this secret key for production
app.secret_key = 'CHANGE_THIS_TO_A_SECURE_RANDOM_KEY'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'loan_model.pkl')
HISTORY_FILE = os.path.join(BASE_DIR, 'data', 'loan_history.json')

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print('Model not loaded:', e)


def ensure_history_file():
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump([], f)


def load_history():
    ensure_history_file()
    with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def append_history(entry):
    history = load_history()
    history.append(entry)
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

# Simple in-memory user store (replace with real auth in production)
USERS = {
    'admin': 'password'
}


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = (request.form.get('username') or '').strip()
        password = request.form.get('password') or ''
        # allow case-insensitive username and ignore surrounding whitespace
        lookup = {k.lower(): k for k in USERS.keys()}
        key = lookup.get(username.lower())
        if key and USERS.get(key) == password:
            session['user'] = key
            return redirect(url_for('dashboard'))
        flash('Invalid username or password. Tip: try username "admin" and password "password"', 'error')
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))


@app.route('/dashboard')
def dashboard():
    if not session.get('user'):
        return redirect(url_for('login'))
    user = session.get('user')
    history = load_history()
    total = len(history)
    approved = sum(1 for item in history if item.get('result') == 'Approved')
    rejected = sum(1 for item in history if item.get('result') == 'Rejected')
    return render_template('dashboard.html', user=user, total=total, approved=approved, rejected=rejected, history=history)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # collect form values
        data = {
            'Gender': request.form.get('gender'),
            'Married': request.form.get('married'),
            'Dependents': request.form.get('dependents'),
            'Education': request.form.get('education'),
            'Self_Employed': request.form.get('self_employed', 'No'),
            'ApplicantIncome': float(request.form.get('applicant_income') or 0),
            'CoapplicantIncome': float(request.form.get('coapplicant_income') or 0),
            'LoanAmount': float(request.form.get('loan_amount') or 0),
            'Loan_Amount_Term': float(request.form.get('loan_term') or 360),
            'Credit_History': int(request.form.get('credit_history') or 1),
            'Property_Area': request.form.get('property_area','Urban')
        }
        df = pd.DataFrame([data])
        if model is None:
            return 'Model not trained. Run python train_model.py first.'
        pred = int(model.predict(df)[0])
        # compute probability for the predicted class if available
        prob = None
        try:
            probs = model.predict_proba(df)[0]
            prob = float(probs[pred])
        except Exception:
            prob = None

        # simple rule-based override for obvious high-risk cases
        high_risk = False
        try:
            if data.get('Credit_History') == 0:
                high_risk = True
            if data.get('ApplicantIncome', 0) < 1500 and data.get('LoanAmount', 0) > 150:
                high_risk = True
            if data.get('Self_Employed') and data.get('Self_Employed').lower() == 'yes' and data.get('ApplicantIncome', 0) < 1200:
                high_risk = True
        except Exception:
            high_risk = False

        if high_risk:
            pred = 0

        result = 'Approved' if pred == 1 else 'Rejected'
        status_class = 'approved' if pred == 1 else 'rejected'
        record = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            **data,
            'result': result,
            'probability': round(prob * 100, 2) if prob is not None else None
        }
        append_history(record)
        return render_template('submit.html', result=result, status_class=status_class, record=record)
    return render_template('predict.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
