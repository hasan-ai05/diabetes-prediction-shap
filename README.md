# 🩺 Diabetes Prediction with SHAP Explainability

> **Predicting diabetes risk from clinical data — and explaining every decision the model makes.**

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Best%20Model-orange?logo=xgboost)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-green)
![Dataset](https://img.shields.io/badge/Dataset-100K%20Patients-lightgrey)
![ROC AUC](https://img.shields.io/badge/ROC--AUC-0.977-brightgreen)

---

## Project Overview

This project builds a machine learning pipeline to **predict whether a patient has diabetes** using 9 clinical features — and goes beyond accuracy by using **SHAP (SHapley Additive exPlanations)** to explain *why* the model made each prediction.

The key insight: a model that can't explain itself is a model you can't trust in healthcare.

---

## Objectives

- Train and compare multiple ML models on a real-world medical dataset
- Identify the most clinically relevant features for diabetes prediction
- Use **SHAP** to make the model interpretable at both the global and individual patient level
- Evaluate performance with metrics appropriate for imbalanced medical data

---

## Dataset

**Source:** [Diabetes Prediction Dataset 2023 — Kaggle](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)

| Property | Value |
|---|---|
| Records | 100,000 patients (96,146 after deduplication) |
| Features | 9 clinical variables |
| Target | Binary — `1` = Diabetic, `0` = Not Diabetic |
| Class balance | 91.2% healthy / 8.8% diabetic (10.3:1 ratio) |

### Feature Descriptions

| Feature | Type | Clinical Meaning |
|---|---|---|
| `gender` | Categorical | Male / Female / Other |
| `age` | Numeric | Patient age in years |
| `hypertension` | Binary | 1 = has hypertension |
| `heart_disease` | Binary | 1 = has heart disease |
| `smoking_history` | Categorical | never / current / former / ever / No Info |
| `bmi` | Numeric | Body mass index (kg/m²) |
| `HbA1c_level` | Numeric | Avg blood sugar over 3 months — **gold standard marker** |
| `blood_glucose_level` | Numeric | Fasting blood glucose (mg/dL) |
| `diabetes` | Binary | **Target variable** |

> **Clinical Note:** HbA1c ≥ 6.5 and blood glucose ≥ 126 mg/dL are the gold-standard clinical thresholds for diabetes diagnosis. This is why these two features dominate model predictions.

---

## Project Structure

```
diabetes-prediction/
│
├── Diabetes_Prediction.ipynb   # Full notebook — EDA, models, SHAP
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

---

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Visualized feature distributions by diagnosis (box plots, histograms)
- Identified HbA1c and blood glucose as the strongest separators
- Discovered former smokers show higher diabetes rates than current smokers — driven by age accumulation, not smoking alone
- Found hypertension present in ~24% of diabetic patients vs ~6% of healthy patients

### 2. Preprocessing
- Label-encoded categorical features (`gender`, `smoking_history`)
- Applied `StandardScaler` — fitted only on training data to prevent data leakage
- Stratified train/test split (80/20) to preserve class ratio in both sets

### 3. Model Training
Three models were trained and compared:

| Model | Description |
|---|---|
| Logistic Regression | Linear baseline — fast and interpretable |
| Random Forest | Ensemble of 200 decision trees — handles non-linearity |
| XGBoost | Gradient boosted trees — state-of-the-art for tabular data |

### 4. Evaluation
Because the dataset is heavily imbalanced (10:1), **accuracy is misleading**. Evaluation focused on:
- **F1 Score** — balances precision and recall
- **ROC-AUC** — evaluates performance across all decision thresholds
- **Recall** — most critical in medical screening (missing a diabetic patient is dangerous)

### 5. SHAP Explainability
- **Global SHAP summary plot** — ranked feature importance across all patients
- **Individual waterfall plot** — explained a single patient's prediction feature by feature

---

## Results

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.9591 | 0.8687 | 0.6321 | 0.7317 | 0.9596 |
| Random Forest | 0.9712 | **0.9856** | 0.6840 | 0.8075 | 0.9727 |
| **XGBoost**  | 0.9708 | 0.9671 | **0.6928** | 0.8073 | **0.9771** |

### 5-Fold Cross Validation

| Model | Mean F1 | Std Dev | Verdict |
|---|---|---|---|
| Logistic Regression | 0.728 | ± 0.010 | Baseline |
| Random Forest | 0.801 | ± 0.007 | Strong |
| **XGBoost** | **0.803** | **± 0.005** | **Most stable** |

**Winner: XGBoost** — highest AUC (0.977) and lowest variance across folds.

### Confusion Matrix — XGBoost

|  | Predicted: Healthy | Predicted: Diabetic |
|---|---|---|
| **Actual: Healthy** | 17,494 ✅ | 40 ❌ |
| **Actual: Diabetic** | 521 ❌ | 1,175 ✅ |

> 521 diabetic patients were missed (False Negatives). Threshold tuning from 0.5 → 0.3 improves recall at the cost of precision — appropriate for a first-line screening tool.

---

##  SHAP Findings

**Global feature importance (ranked by SHAP):**

1.  `HbA1c_level` — dominant predictor, almost alone splits the two groups
2.  `blood_glucose_level` — strong second signal
3.  `age` — meaningful contributor, especially above 50
4. `bmi` — present but weaker than expected
5. `hypertension`, `smoking_history` — secondary signals

**Individual prediction example:**
A 54-year-old female with blood glucose of 280 mg/dL was predicted diabetic with **99.9% probability**. The waterfall plot showed blood glucose and HbA1c as the dominant positive contributors.

---

##  How to Run

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/diabetes-prediction.git
cd diabetes-prediction

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook Diabetes_Prediction.ipynb
```

---

##  Requirements

```
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
xgboost
shap
kagglehub
jupyter
```

---

## Key Takeaways

1. **HbA1c is king** — it reflects 3 months of blood sugar history, not a single reading. SHAP confirmed it as the most influential feature.
2. **ROC-AUC beats F1 for model selection** — F1 evaluates one threshold; AUC evaluates all of them. For medical screening, AUC is the right metric.
3. **Recall matters most in healthcare** — missing a diabetic patient (false negative) has real-world consequences. Threshold tuning is a clinical decision, not just a technical one.
4. **SHAP makes models trustworthy** — a prediction without an explanation is not useful in medicine. SHAP bridges the gap between ML and clinical interpretability.

---

## Author

**Hasan Akhras**  
[LinkedIn](https://linkedin.com/in/YOUR_HANDLE) · [GitHub](https://github.com/YOUR_USERNAME)

---

## License

This project is open source and available under the [MIT License](LICENSE).
