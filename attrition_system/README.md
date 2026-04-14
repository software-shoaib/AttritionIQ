# 🧠 AttritionIQ — Employee Attrition Prediction System

> A machine learning web application that predicts employee attrition risk using **Random Forest** and **SVM**, served via a **Flask** local server.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [ML Models](#ml-models)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the System](#running-the-system)
- [Running on Google Colab](#running-on-google-colab)
- [API Endpoints](#api-endpoints)
- [Features](#features)
- [Model Performance](#model-performance)
- [Dataset](#dataset)

---

## Overview

AttritionIQ analyzes HR employee data and predicts whether an employee is likely to leave the organization. It provides:

- **Single employee** risk prediction with probability score
- **Batch prediction** via CSV upload
- **Feature importance** visualization
- **Model metrics** dashboard (Accuracy + AUC-ROC)
- **Risk classification**: High / Medium / Low

---

## Tech Stack

| Layer | Technology | Reason |
|---|---|---|
| Language | Python 3.10+ | Dominant ML/data science ecosystem |
| ML Framework | scikit-learn | Battle-tested RF, SVM, pipelines |
| API Server | Flask | Lightweight REST API; ideal for local ML serving |
| Frontend | HTML / CSS / JS | Zero build step; rendered via Flask templates |
| Serialization | Pickle | Standard scikit-learn model persistence |
| Data | pandas, numpy | Data preprocessing and manipulation |

---

## ML Models

### Random Forest
- Ensemble of **200 decision trees** (bagging)
- Handles mixed data (categorical + numerical) natively
- Provides **feature importance scores** for HR explainability
- Robust to outliers and noise common in HR survey data
- `class_weight='balanced'` to handle attrition class imbalance

### SVM (RBF Kernel)
- Finds the **optimal maximum-margin decision boundary**
- RBF kernel captures **non-linear attrition patterns**
- Wrapped in a `Pipeline` with `StandardScaler` for proper scaling
- `probability=True` enables Platt scaling for probability output
- Complements Random Forest via ensemble voting

### Ensemble (Soft Voting)
- Combines RF (weight: **0.6**) + SVM (weight: **0.4**)
- Averages predicted probabilities from both models
- Reduces individual model error → more robust risk scores

---

## Project Structure

```
attrition_system/
├── app.py                  # Flask REST API server
├── train_models.py         # Training pipeline (RF + SVM + Ensemble)
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html          # Web dashboard (single-page)
├── models/                 # Auto-generated after training
│   ├── rf_model.pkl
│   ├── svm_model.pkl
│   ├── ensemble_model.pkl
│   ├── encoders.pkl
│   ├── target_encoder.pkl
│   └── meta.json
└── data/
    └── hr_data.csv         # Auto-generated synthetic HR dataset
```

---

## Setup & Installation

### 1. Clone / Extract the project
```bash
cd attrition_system
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the models
```bash
python train_models.py
```

> Models are saved to `/models/`. This step is **automatic** on first server start if models are missing.

---

## Running the System

```bash
python app.py
```

Open your browser and go to:

```
http://localhost:5000
```

---

## Running on Google Colab

### Step 1 — Mount Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 2 — Set project path
```python
import os
os.chdir('/content/drive/MyDrive/attrition_system')
```

### Step 3 — Install dependencies
```python
!pip install flask scikit-learn pandas numpy pyngrok -q
```

### Step 4 — Patch app.py for Colab
```python
with open('app.py', 'r') as f:
    content = f.read()
content = content.replace(
    "app.run(debug=True, host='0.0.0.0', port=5000)",
    "app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)"
)
with open('app.py', 'w') as f:
    f.write(content)
```

### Step 5 — Train models
```python
%run train_models.py
```

### Step 6 — Launch with ngrok
```python
from pyngrok import ngrok
import threading, os

ngrok.set_auth_token("YOUR_REAL_TOKEN_FROM_dashboard.ngrok.com")

public_url = ngrok.connect(5000)
print(f"Open: {public_url}")

thread = threading.Thread(target=lambda: os.system("python app.py"))
thread.daemon = True
thread.start()
```

> Get a free token at [https://dashboard.ngrok.com/get-started/your-authtoken](https://dashboard.ngrok.com/get-started/your-authtoken)

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web dashboard UI |
| `POST` | `/predict` | Single employee prediction (JSON) |
| `POST` | `/batch_predict` | CSV batch upload → predictions CSV |
| `GET` | `/model_metrics` | Accuracy + AUC-ROC for all models |
| `GET` | `/feature_importance` | Top 10 RF feature importances |
| `GET` | `/health` | Server health check |

### Sample Request — `/predict`
```json
POST /predict
Content-Type: application/json

{
  "Age": 28,
  "MonthlyIncome": 3500,
  "OverTime": "Yes",
  "JobSatisfaction": 2,
  "WorkLifeBalance": 1,
  "BusinessTravel": "Travel_Frequently",
  "YearsAtCompany": 2,
  "Department": "Sales",
  "model_choice": "ensemble"
}
```

### Sample Response
```json
{
  "prediction": "Yes",
  "attrition_prob": 74.3,
  "risk_level": "High",
  "model_used": "ensemble"
}
```

---

## Features

| Feature | Description |
|---|---|
| Single Prediction | Enter 12 employee attributes → instant risk score |
| Model Selector | Switch between RF, SVM, or Ensemble per prediction |
| Risk Levels | High (≥65%) / Medium (35–64%) / Low (<35%) |
| Probability Bar | Visual bar showing attrition probability |
| Batch Prediction | Upload CSV → download results with 3 new columns |
| Feature Importance | Top 10 most influential HR factors (RF-based) |
| Model Metrics | Side-by-side Accuracy + AUC-ROC for all 3 models |

---

## Model Performance

Trained on 1,500 synthetic IBM-style HR records (80/20 split):

| Model | Accuracy | AUC-ROC |
|---|---|---|
| Random Forest | 83.67% | 0.6670 |
| SVM (RBF) | 77.00% | 0.5940 |
| **Ensemble** | **83.67%** | **0.6674** |

**Top Attrition Predictors (Feature Importance):**
1. Monthly Income
2. OverTime
3. Years At Company
4. Daily Rate
5. Total Working Years

---

## Dataset

A synthetic IBM HR Analytics-style dataset is **auto-generated** on first run (1,500 records, ~16.5% attrition rate). To use your own dataset, place a CSV file at `data/hr_data.csv` with the required columns before running `train_models.py`.

Required columns: `Age`, `BusinessTravel`, `Department`, `DistanceFromHome`, `Education`, `EducationField`, `EnvironmentSatisfaction`, `Gender`, `HourlyRate`, `JobInvolvement`, `JobLevel`, `JobRole`, `JobSatisfaction`, `MaritalStatus`, `MonthlyIncome`, `MonthlyRate`, `NumCompaniesWorked`, `OverTime`, `PercentSalaryHike`, `PerformanceRating`, `RelationshipSatisfaction`, `StockOptionLevel`, `TotalWorkingYears`, `TrainingTimesLastYear`, `WorkLifeBalance`, `YearsAtCompany`, `YearsInCurrentRole`, `YearsSinceLastPromotion`, `YearsWithCurrManager`, `Attrition`

---

> Built with Python · Flask · scikit-learn | Iqra University — Software Engineering
