"""
Employee Attrition Prediction - Model Training
==============================================
Models: Random Forest + SVM (with ensemble voting)
Dataset: IBM HR Analytics (synthetic generation if not available)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score
)
from sklearn.pipeline import Pipeline
import pickle
import json
import os
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. GENERATE / LOAD DATASET
# ─────────────────────────────────────────────

def generate_hr_dataset(n=1500, seed=42):
    """Generates a realistic synthetic IBM-style HR dataset."""
    np.random.seed(seed)
    departments   = ['Sales', 'Research & Development', 'Human Resources']
    job_roles     = ['Sales Executive', 'Research Scientist', 'Laboratory Technician',
                     'Manufacturing Director', 'Healthcare Representative', 'Manager',
                     'Sales Representative', 'Research Director', 'Human Resources']
    edu_fields    = ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree',
                     'Human Resources', 'Other']
    marital_stat  = ['Single', 'Married', 'Divorced']

    df = pd.DataFrame({
        'Age':                    np.random.randint(18, 60, n),
        'BusinessTravel':         np.random.choice(['Non-Travel','Travel_Rarely','Travel_Frequently'], n, p=[0.3,0.5,0.2]),
        'DailyRate':              np.random.randint(100, 1500, n),
        'Department':             np.random.choice(departments, n),
        'DistanceFromHome':       np.random.randint(1, 30, n),
        'Education':              np.random.randint(1, 6, n),
        'EducationField':         np.random.choice(edu_fields, n),
        'EnvironmentSatisfaction':np.random.randint(1, 5, n),
        'Gender':                 np.random.choice(['Male','Female'], n),
        'HourlyRate':             np.random.randint(30, 100, n),
        'JobInvolvement':         np.random.randint(1, 5, n),
        'JobLevel':               np.random.randint(1, 6, n),
        'JobRole':                np.random.choice(job_roles, n),
        'JobSatisfaction':        np.random.randint(1, 5, n),
        'MaritalStatus':          np.random.choice(marital_stat, n),
        'MonthlyIncome':          np.random.randint(1000, 20000, n),
        'MonthlyRate':            np.random.randint(2000, 27000, n),
        'NumCompaniesWorked':     np.random.randint(0, 10, n),
        'OverTime':               np.random.choice(['Yes','No'], n, p=[0.3,0.7]),
        'PercentSalaryHike':      np.random.randint(11, 25, n),
        'PerformanceRating':      np.random.randint(3, 5, n),
        'RelationshipSatisfaction':np.random.randint(1, 5, n),
        'StockOptionLevel':       np.random.randint(0, 4, n),
        'TotalWorkingYears':      np.random.randint(0, 40, n),
        'TrainingTimesLastYear':  np.random.randint(0, 7, n),
        'WorkLifeBalance':        np.random.randint(1, 5, n),
        'YearsAtCompany':         np.random.randint(0, 40, n),
        'YearsInCurrentRole':     np.random.randint(0, 20, n),
        'YearsSinceLastPromotion':np.random.randint(0, 15, n),
        'YearsWithCurrManager':   np.random.randint(0, 17, n),
    })

    # Simulate realistic attrition (correlated with OT, satisfaction, travel)
    attrition_prob = (
        0.05
        + 0.20 * (df['OverTime'] == 'Yes')
        + 0.10 * (df['JobSatisfaction'] <= 2)
        + 0.08 * (df['BusinessTravel'] == 'Travel_Frequently')
        + 0.06 * (df['WorkLifeBalance'] <= 2)
        - 0.05 * (df['JobLevel'] >= 4)
        - 0.04 * (df['YearsAtCompany'] > 10)
    ).clip(0.02, 0.85)

    df['Attrition'] = np.where(np.random.rand(n) < attrition_prob, 'Yes', 'No')
    return df


# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────

CATEGORICAL_COLS = [
    'BusinessTravel', 'Department', 'EducationField',
    'Gender', 'JobRole', 'MaritalStatus', 'OverTime'
]

def preprocess(df, encoders=None, fit=True):
    df = df.copy()
    if encoders is None:
        encoders = {}

    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            df[col] = df[col].astype(str).map(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

    return df, encoders


# ─────────────────────────────────────────────
# 3. TRAIN MODELS
# ─────────────────────────────────────────────

def train():
    print("=" * 55)
    print("  Employee Attrition Prediction — Model Training")
    print("=" * 55)

    # Load or generate data
    data_path = 'data/hr_data.csv'
    if os.path.exists(data_path):
        print(f"[+] Loading dataset from {data_path}")
        df = pd.read_csv(data_path)
    else:
        print("[+] Generating synthetic HR dataset (1500 records)...")
        df = generate_hr_dataset(1500)
        os.makedirs('data', exist_ok=True)
        df.to_csv(data_path, index=False)
        print(f"[+] Dataset saved to {data_path}")

    print(f"    Shape: {df.shape}  |  Attrition rate: {(df['Attrition']=='Yes').mean()*100:.1f}%")

    # Encode target
    target_enc = LabelEncoder()
    y = target_enc.fit_transform(df['Attrition'])  # Yes=1, No=0

    # Drop target + irrelevant cols
    X_raw = df.drop(columns=['Attrition'], errors='ignore')

    # Preprocess
    X_proc, encoders = preprocess(X_raw, fit=True)

    feature_names = X_proc.columns.tolist()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_proc, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n[+] Train: {len(X_train)} | Test: {len(X_test)}")

    # ── Random Forest ──────────────────────────
    print("\n[*] Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc  = accuracy_score(y_test, rf_pred)
    rf_auc  = roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])
    print(f"    Accuracy: {rf_acc*100:.2f}%  |  AUC-ROC: {rf_auc:.4f}")

    # ── SVM ───────────────────────────────────
    print("\n[*] Training SVM (RBF kernel)...")
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            probability=True,
            class_weight='balanced',
            random_state=42
        ))
    ])
    svm_pipeline.fit(X_train, y_train)
    svm_pred = svm_pipeline.predict(X_test)
    svm_acc  = accuracy_score(y_test, svm_pred)
    svm_auc  = roc_auc_score(y_test, svm_pipeline.predict_proba(X_test)[:,1])
    print(f"    Accuracy: {svm_acc*100:.2f}%  |  AUC-ROC: {svm_auc:.4f}")

    # ── Ensemble (Soft Voting) ─────────────────
    print("\n[*] Training Ensemble (Soft Voting: RF + SVM)...")
    ensemble = VotingClassifier(
        estimators=[
            ('rf',  rf),
            ('svm', svm_pipeline)
        ],
        voting='soft',
        weights=[0.6, 0.4]   # RF slightly more weight (better on tabular)
    )
    ensemble.fit(X_train, y_train)
    ens_pred = ensemble.predict(X_test)
    ens_acc  = accuracy_score(y_test, ens_pred)
    ens_auc  = roc_auc_score(y_test, ensemble.predict_proba(X_test)[:,1])
    print(f"    Accuracy: {ens_acc*100:.2f}%  |  AUC-ROC: {ens_auc:.4f}")

    # ── Classification Report ─────────────────
    print("\n[+] Ensemble Classification Report:")
    print(classification_report(y_test, ens_pred, target_names=target_enc.classes_))

    # ── Feature Importance (RF) ───────────────
    importances = dict(zip(feature_names, rf.feature_importances_))
    top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\n[+] Top 10 Feature Importances (Random Forest):")
    for feat, imp in top_features:
        bar = '█' * int(imp * 100)
        print(f"    {feat:<35} {imp:.4f}  {bar}")

    # ── Save Artifacts ────────────────────────
    os.makedirs('models', exist_ok=True)

    with open('models/rf_model.pkl',       'wb') as f: pickle.dump(rf, f)
    with open('models/svm_model.pkl',      'wb') as f: pickle.dump(svm_pipeline, f)
    with open('models/ensemble_model.pkl', 'wb') as f: pickle.dump(ensemble, f)
    with open('models/encoders.pkl',       'wb') as f: pickle.dump(encoders, f)
    with open('models/target_encoder.pkl', 'wb') as f: pickle.dump(target_enc, f)

    meta = {
        'feature_names': feature_names,
        'categorical_cols': CATEGORICAL_COLS,
        'metrics': {
            'random_forest': {'accuracy': round(rf_acc,4), 'auc_roc': round(rf_auc,4)},
            'svm':           {'accuracy': round(svm_acc,4),'auc_roc': round(svm_auc,4)},
            'ensemble':      {'accuracy': round(ens_acc,4),'auc_roc': round(ens_auc,4)},
        },
        'top_features': [{'feature': f, 'importance': round(i,4)} for f,i in top_features],
        'attrition_rate': round((df['Attrition']=='Yes').mean()*100, 1),
        'dataset_size': len(df)
    }
    with open('models/meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print("\n[✓] All models + metadata saved to /models/")
    print("=" * 55)
    return meta


if __name__ == '__main__':
    train()
