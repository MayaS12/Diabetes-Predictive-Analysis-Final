Predicting the Comeback: Diabetes and Hospital Readmissions
Authors: Maya Shah and Jessica Phan
Dataset: UCI Diabetes 130-US hospitals for years 1999–2008

Overview
This project explores machine learning approaches to predict the likelihood of hospital readmission for patients with diabetes. In 2011 alone, over $41 billion was spent by American hospitals on diabetic patient readmissions within 30 days. By identifying key predictive factors, we aim to help reduce healthcare costs and improve the quality of care.

The model analyzes a variety of features, including:

Duration of hospital stay

Age and weight

Number of procedures and medications

Primary and secondary diagnoses

Goals
Identify the most influential factors contributing to early hospital readmissions.

Compare and evaluate several machine learning models:

Logistic Regression

Decision Tree

Random Forest

XGBoost

Use SMOTE to address class imbalance and enhance model robustness.

Select the best-performing model (Random Forest) for final deployment.

Tools and Libraries
numpy

pandas

scikit-learn

xgboost

seaborn

matplotlib

IPython

imblearn (for SMOTE)

Methodology
Data Preprocessing

Cleaned and filtered relevant columns

Handled missing values

Encoded categorical variables

Applied SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance

Modeling and Evaluation

Trained and evaluated the following models:

Logistic Regression

Decision Tree

Random Forest

XGBoost

Assessed performance using:

Accuracy

Precision

Recall

F1-score

ROC-AUC Score

Model Selection

Random Forest demonstrated the best overall performance across evaluation metrics.

Results
Random Forest achieved the highest balance of precision and recall on the test set.

XGBoost also performed competitively but was slightly outperformed by Random Forest.

Feature importance analysis highlighted top predictors of readmission, such as number of inpatient visits and time in hospital.

SMOTE significantly improved performance on the minority class (readmitted within 30 days).

Dataset
Source: UCI Machine Learning Repository

Description: Medical records from 130 US hospitals (1999–2008), with over 50 features pertaining to diabetic patients, including diagnoses, medications, lab results, and admission details.

Future Work
Perform hyperparameter tuning with GridSearchCV for XGBoost and Random Forest

Explore additional ensemble techniques and feature engineering

Incorporate demographic or socioeconomic data for deeper insights

Develop a real-time prediction dashboard for clinical use

Contact
For questions or collaboration opportunities:

Maya Shah

Jessica Phan

