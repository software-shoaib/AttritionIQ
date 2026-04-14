# Employee Attrition Prediction System 🚀

## Project Discussion

## Overview
The system predicts whether an employee is likely to leave an organization using machine learning. It is deployed as a local web application, enabling users to interact with the model through a simple interface.

---

## Models Used
Two complementary machine learning algorithms were implemented:

- **Random Forest**
  - Handles mixed HR data (categorical + numerical)
  - Provides feature importance scores
  - Uses an ensemble of 200 decision trees for stable predictions

- **Support Vector Machine (SVM - RBF Kernel)**
  - Finds optimal decision boundaries in moderate-sized datasets
  - Captures non-linear attrition patterns effectively

### Ensemble Approach
Both models are combined into a **soft-voting ensemble**:
- Random Forest → 60% weight  
- SVM → 40% weight  

This ensures maximum reliability and improved prediction performance.

---

## Tech Stack
- **Python** – Core programming language  
- **Scikit-learn** – Model training and preprocessing  
- **Pickle** – Model serialization  
- **Flask** – Lightweight REST API backend (5 endpoints)  
- **HTML / CSS / JavaScript** – Frontend dashboard  

The frontend is a **single-page application** rendered directly through Flask templates, with no separate build step required.

---

## Key Features
- **Single Employee Prediction**
  - Instant risk scoring: High / Medium / Low  

- **Batch Prediction**
  - Upload CSV files to analyze entire workforces  

- **Feature Importance Visualization**
  - Displays top 10 attrition drivers  

- **Model Metrics Dashboard**
  - Shows Accuracy and AUC-ROC for all models  

---

## Results Achieved
- Dataset: Synthetic IBM-style HR dataset  
- Total Records: 1,500  
- Attrition Rate: 16.5%  

### Performance
- Accuracy: **83.67%**  
- AUC-ROC: **0.667**

### Top Predictors
- Monthly Income  
- Overtime  
- Years at Company  
- Age  
- Distance from Home  

---

## Deployment
The system runs on a local Flask server:

```bash
python app.py
```

### WEb UI

<img src="https://github.com/software-shoaib/AttritionIQ/blob/main/1.png">
<<<<<<< HEAD
<img src="https://github.com/software-shoaib/AttritionIQ/blob/main/2.png">
<img src="https://github.com/software-shoaib/AttritionIQ/blob/main/3.png">
<img src="https://github.com/software-shoaib/AttritionIQ/blob/main/4.png">
<img src="https://github.com/software-shoaib/AttritionIQ/blob/main/5.png">
=======
>>>>>>> f97f88c04886b2e877bd648c566340cc0d49db00
