# Predicting the Comeback: Diabetes and Hospital Readmissions

**Authors:** Maya Shah · Jessica Phan  
**Dataset:** [UCI Diabetes 130-US hospitals for years 1999–2008](https://archive.ics.uci.edu/dataset/34/diabetes)

---

## Overview

Hospital readmissions within 30 days are a major cost driver for diabetes care—exceeding **$41 billion in 2011** alone. This project applies supervised machine learning techniques to predict whether a patient with diabetes will be readmitted within 30 days of discharge. By identifying high-risk cases in advance, hospitals can target follow-up interventions that improve outcomes and reduce costs.

Key features analyzed include:

- Length of hospital stay
- Age and weight
- Number of procedures and medications
- Primary and secondary diagnoses

---

## Goals

- Identify the most influential factors contributing to early readmissions
- Train and evaluate multiple machine learning models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - XGBoost
- Address class imbalance using **SMOTE**
- Select the best-performing model (**Random Forest**) for deployment

---

## Tools & Libraries

- Python 3.10+
- `numpy`
- `pandas`
- `scikit-learn`
- `xgboost`
- `imblearn` (SMOTE)
- `seaborn`
- `matplotlib`
- `IPython`

---

## Methodology

### Data Preprocessing

- Cleaned and filtered relevant columns
- Handled missing values
- Encoded categorical variables using one-hot encoding
- Applied SMOTE to balance the dataset

### Model Training & Evaluation

- Models trained:

  - Logistic Regression
  - Decision Tree
  - Random Forest
  - XGBoost

- Evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC

### Model Selection

- Random Forest provided the highest overall performance and interpretability

---

## Results

- Random Forest achieved the highest balance of precision and recall on the test set.

- XGBoost also performed competitively but was slightly outperformed by Random Forest.

- Feature importance analysis highlighted top predictors of readmission, such as number of inpatient visits and time in hospital.

- SMOTE significantly improved performance on the minority class (readmitted within 30 days).

**Top predictive features (from Random Forest):**

- Number of inpatient visits
- Time in hospital
- Diagnosis codes

---

## Dataset

- **Source:** UCI Machine Learning Repository
- **Records:** 101,766 patient encounters
- **Years Covered:** 1999–2008
- **Attributes:** 50+ features including lab results, medications, diagnoses, and administrative data
- **Target:** Binary label — readmission within 30 days vs. not

---

## Future Work

- Hyperparameter tuning with `GridSearchCV`
- Integrate LightGBM and CatBoost for performance comparison
- Add external demographic/socioeconomic data
- Build a clinical dashboard for real-time risk scoring

---

## Contact

For questions or collaboration:

- **Maya Shah**
- **Jessica Phan**
