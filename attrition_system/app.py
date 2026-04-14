"""
Employee Attrition Prediction — Flask API Server
=================================================
Endpoints:
  GET  /                  → Dashboard (HTML)
  POST /predict           → Single employee prediction
  POST /batch_predict     → CSV batch prediction
  GET  /model_metrics     → Model performance stats
  GET  /feature_importance→ Top features JSON
"""

from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import pickle
import json
import os
import io
import warnings
warnings.filterwarnings('ignore')

from train_models import preprocess, CATEGORICAL_COLS

app = Flask(__name__)

# ─────────────────────────────────────────────
# LOAD MODELS ON STARTUP
# ─────────────────────────────────────────────

def load_models():
    models = {}
    paths = {
        'rf':       'models/rf_model.pkl',
        'svm':      'models/svm_model.pkl',
        'ensemble': 'models/ensemble_model.pkl',
        'encoders': 'models/encoders.pkl',
        'target':   'models/target_encoder.pkl',
    }
    for key, path in paths.items():
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models[key] = pickle.load(f)
            print(f"  [✓] Loaded: {path}")
        else:
            print(f"  [!] Missing: {path} — run train_models.py first")

    with open('models/meta.json') as f:
        models['meta'] = json.load(f)

    return models


MODELS = {}

@app.before_request
def ensure_models():
    global MODELS
    if not MODELS:
        MODELS = load_models()


# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────

def predict_employee(data: dict, model_choice='ensemble'):
    df = pd.DataFrame([data])

    # Fill missing numeric columns with 0
    meta = MODELS['meta']
    for col in meta['feature_names']:
        if col not in df.columns:
            df[col] = 0

    df = df[meta['feature_names']]

    # Preprocess
    df_proc, _ = preprocess(df, encoders=MODELS['encoders'], fit=False)

    model = MODELS.get(model_choice, MODELS['ensemble'])

    pred_class = model.predict(df_proc)[0]
    pred_proba = model.predict_proba(df_proc)[0]

    label = MODELS['target'].inverse_transform([pred_class])[0]
    attrition_prob = float(pred_proba[1])  # probability of "Yes"

    risk_level = (
        'High'   if attrition_prob >= 0.65 else
        'Medium' if attrition_prob >= 0.35 else
        'Low'
    )

    return {
        'prediction':      label,
        'attrition_prob':  round(attrition_prob * 100, 1),
        'risk_level':      risk_level,
        'model_used':      model_choice
    }


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html', meta=MODELS.get('meta', {}))


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON body provided'}), 400

        model_choice = data.pop('model_choice', 'ensemble')
        result = predict_employee(data, model_choice)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        f = request.files['file']
        df = pd.read_csv(f)
        model_choice = request.form.get('model_choice', 'ensemble')

        results = []
        for _, row in df.iterrows():
            try:
                r = predict_employee(row.to_dict(), model_choice)
                results.append({**row.to_dict(), **r})
            except Exception as e:
                results.append({'error': str(e)})

        out_df = pd.DataFrame(results)
        buf = io.StringIO()
        out_df.to_csv(buf, index=False)
        buf.seek(0)
        return send_file(
            io.BytesIO(buf.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name='attrition_predictions.csv'
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/model_metrics')
def model_metrics():
    meta = MODELS.get('meta', {})
    return jsonify(meta.get('metrics', {}))


@app.route('/feature_importance')
def feature_importance():
    meta = MODELS.get('meta', {})
    return jsonify(meta.get('top_features', []))


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'models_loaded': bool(MODELS)})


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "="*55)
    print("  Employee Attrition Prediction System")
    print("  Flask Server  ·  http://localhost:5000")
    print("="*55 + "\n")

    # Auto-train if models don't exist
    if not os.path.exists('models/ensemble_model.pkl'):
        print("[!] Models not found. Running training first...\n")
        from train_models import train
        train()

    MODELS = load_models()
    app.run(debug=True, host='0.0.0.0', port=5000)
