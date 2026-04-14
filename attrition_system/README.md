# AttritionIQ — Employee Attrition Prediction System

## Tech Stack & Rationale

| Component | Tech | Why |
|-----------|------|-----|
| Language | Python 3.10+ | Dominant in ML/data science; rich ecosystem |
| ML Models | Random Forest + SVM | Complementary strengths (see below) |
| API/Server | Flask | Lightweight REST API; easy pickle model integration |
| Frontend | HTML/CSS/JS | No build step; runs directly from Flask templates |
| Serialization | Pickle | Standard for scikit-learn model persistence |

## Why Random Forest?
- Handles mixed categorical + numerical data natively
- Provides feature importance scores → explainable HR insights
- Robust to outliers/noise common in HR surveys
- Ensemble of 200 decision trees → low variance predictions
- No need for feature scaling

## Why SVM?
- Excels at finding optimal decision boundary with max margin
- RBF kernel captures non-linear attrition patterns
- Works well on moderate-sized datasets (< 100k rows)
- Complements Random Forest via soft-voting ensemble
- Probability calibration via Platt scaling (probability=True)

## Why Ensemble (Voting)?
- RF weight: 0.6 | SVM weight: 0.4
- Combines diverse models → reduces error
- Soft voting averages probabilities → smoother risk scores

---

## Setup Instructions

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train models (auto-runs on first server start)
```bash
python train_models.py
```

### 3. Start Flask server
```bash
python app.py
```

### 4. Open browser
```
http://localhost:5000
```

---

## Project Structure
```
attrition_system/
├── app.py              # Flask API server
├── train_models.py     # Training pipeline (RF + SVM + Ensemble)
├── requirements.txt
├── models/             # Saved .pkl files + meta.json (auto-generated)
├── data/               # HR dataset CSV (auto-generated)
└── templates/
    └── index.html      # Web dashboard
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET  | `/` | Dashboard UI |
| POST | `/predict` | Single employee prediction (JSON) |
| POST | `/batch_predict` | CSV batch upload → downloadable results |
| GET  | `/model_metrics` | Accuracy + AUC-ROC for all models |
| GET  | `/feature_importance` | Top 10 RF feature importances |
| GET  | `/health` | Server health check |

### Sample `/predict` Request
```json
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

## Risk Levels
| Risk | Probability | Action |
|------|-------------|--------|
| 🔴 High | ≥ 65% | Immediate retention intervention |
| 🟡 Medium | 35–64% | Monitor + engagement programs |
| 🟢 Low | < 35% | Standard HR touchpoints |
