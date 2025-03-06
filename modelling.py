import IPython
import pandas as pd
import numpy as np
from statistics import mode
import statsmodels.api as sm
import matplotlib.pyplot as plt  
from sklearn.preprocessing import MinMaxScaler
import scipy as sp
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image
import scipy.stats as stats
import pydotplus
from sklearn import tree


# Load modified file with some pre-processing completed
df = pd.read_csv("diabetes_data_preprocessed.csv", index_col=0)

# Convert data type of nominal features to 'object' type (to handle categorical data properly)
i = ['encounter_id', 'patient_nbr', 'gender', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id', 
     'A1Cresult', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 
     'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 
     'tolazamide', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 
     'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed', 'age', 
     'max_glu_serum', 'level1_diag1', 'level1_diag2', 'level1_diag3', 'level2_diag1', 'level2_diag2', 'level2_diag3']

df[i] = df[i].astype('object')

# Generate two random integer arrays (for illustration, no need for this in production code)
L1 = np.random.randint(1, 10, 20)
L2 = np.random.randint(1, 20, 20)

# Create a new dataframe for the correlation example
datframe = pd.DataFrame({'L1': L1, 'L2': L2})

# Scale data using MinMaxScaler to normalize the values between 0 and 1
scaler = MinMaxScaler()
datframe_scaled = pd.DataFrame(scaler.fit_transform(datframe), columns=['L1', 'L2'])

# Convert 'age' column back to integer type (for consistency in data processing)
df['age'] = df['age'].astype('int64')

# Convert 'age' categories to mid-point values (this could be simplified with a direct mapping)
age_dict = {1: 5, 2: 15, 3: 25, 4: 35, 5: 45, 6: 55, 7: 65, 8: 75, 9: 85, 10: 95}
df['age'] = df.age.map(age_dict)

# Skewness and Kurtosis adjustments: Detecting features with high skewness and kurtosis for log transformation
num_col = df.select_dtypes(include='number').columns  # Dynamically select numeric columns from dataframe

# Initialize lists to store calculated values
skew_before, skew_after, kurt_before, kurt_after = [], [], [], []
standard_deviation_before, standard_deviation_after = [], []
log_transform_needed, log_type = [], []

# Iterate over each numeric column and calculate skewness, kurtosis, and apply transformations if necessary
for column in num_col:
    # Get skewness, kurtosis, and standard deviation before transformation
    skew_before_val = df[column].skew()
    kurt_before_val = df[column].kurtosis()
    std_before_val = df[column].std()

    skew_before.append(skew_before_val)
    kurt_before.append(kurt_before_val)
    standard_deviation_before.append(std_before_val)

    # If skewness and kurtosis are both high, log transformation may be needed
    if abs(skew_before_val) > 2 and abs(kurt_before_val) > 2:
        log_transform_needed.append('Yes')

        # Check if values are near zero, if so apply log1p to avoid log(0) issues
        if (df[column] == 0).mean() <= 0.02:
            log_type.append('log')
            transformed_data = np.log(df[df[column] > 0][column])

            skew_after_val = transformed_data.skew()
            kurt_after_val = transformed_data.kurtosis()
            std_after_val = transformed_data.std()
        else:
            log_type.append('log1p')
            transformed_data = np.log1p(df[df[column] >= 0][column])

            skew_after_val = transformed_data.skew()
            kurt_after_val = transformed_data.kurtosis()
            std_after_val = transformed_data.std()
    else:
        log_transform_needed.append('No')
        log_type.append('NA')

        # Keep original skewness and kurtosis values when no transformation is needed
        skew_after_val = skew_before_val
        kurt_after_val = kurt_before_val
        std_after_val = std_before_val

    # Append the post-transformation values
    skew_after.append(skew_after_val)
    kurt_after.append(kurt_after_val)
    standard_deviation_after.append(std_after_val)

# Create a dataframe to store the results of skewness, kurtosis, standard deviation before and after transformations
statdataframe = pd.DataFrame({
    'numeric_column': num_col,
    'skew_before': skew_before,
    'kurtosis_before': kurt_before,
    'standard_deviation_before': standard_deviation_before,
    'log_transform_needed': log_transform_needed,
    'log_type': log_type,
    'skew_after': skew_after,
    'kurtosis_after': kurt_after,
    'standard_deviation_after': standard_deviation_after
})

# Perform log transformation for the columns identified as needing transformation
for i in range(len(statdataframe)):
    if statdataframe['log_transform_needed'][i] == 'Yes':
        colname = str(statdataframe['numeric_column'][i])
        
        # Apply log transformation (log for positive values, log1p for non-negative values)
        if statdataframe['log_type'][i] == 'log':
            # Filter out rows where column values are <= 0 (log cannot handle 0 or negative values)
            df = df[df[colname] > 0]
            df[colname + "_log"] = np.log(df[colname])
            
        elif statdataframe['log_type'][i] == 'log1p':
            # For log1p, filter out rows where column values are negative
            df = df[df[colname] >= 0]
            df[colname + "_log1p"] = np.log1p(df[colname])

# Drop columns related to service utilization (assuming these are irrelevant to model)
df = df.drop(['number_outpatient', 'number_inpatient', 'number_emergency','service_utilization'], axis=1)

numerics = list(set(list(df._get_numeric_data().columns))- {'readmitted'})

# Convert categorical features into integer type for correct processing
df.encounter_id = df.encounter_id.astype('int64')
df.patient_nbr = df.patient_nbr.astype('int64')
df.diabetesMed = df.diabetesMed.astype('int64')
df.change = df.change.astype('int64')

# Convert nominal (binary/categorical) features to integer type for easier aggregation
i = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 
     'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 
     'troglitazone', 'tolazamide', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 
     'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone','A1Cresult']
df[i] = df[i].astype('int64')

# Create a copy of the DataFrame to preserve the original structure (redundant line - will remove one)
dfcopy = df.copy(deep=True)

# Reassign df to dfcopy - redundant, dfcopy is already a full copy of df
df = dfcopy.copy(deep=True)

# Replace 'readmitted' value '2' with '0' (assuming '2' represents not readmitted)
df['readmitted'] = df['readmitted'].apply(lambda x: 0 if x == 2 else x)

# Drop columns related to diagnosis (granular disease information) that are unnecessary for model
df.drop(['diag_1', 'diag_2', 'diag_3', 'level2_diag1', 'level1_diag2', 'level2_diag2', 'level1_diag3',
         'level2_diag3'], axis=1, inplace=True)

# Define a list of interaction terms to create new features
interactionterms = [('num_medications','time_in_hospital'),
                    ('num_medications','num_procedures'),
                    ('time_in_hospital','num_lab_procedures'),
                    ('num_medications','num_lab_procedures'),
                    ('num_medications','number_diagnoses'),
                    ('age','number_diagnoses'),
                    ('change','num_medications'),
                    ('number_diagnoses','time_in_hospital'),
                    ('num_medications','numchange')]

# Generate interaction terms by multiplying the relevant columns
for inter in interactionterms:
    name = inter[0] + '|' + inter[1]
    df[name] = df[inter[0]] * df[inter[1]]

# Create a DataFrame to store statistical values for numeric features
stat_df = pd.DataFrame()
stat_df['features'] = numerics
stat_df['std_dev'] = stat_df['features'].apply(lambda x: df[x].std())  # Compute standard deviation
stat_df['mean'] = stat_df['features'].apply(lambda x: df[x].mean())  # Compute mean

# Logical order: First remove duplicates, then handle outliers, and finally apply scaling

# Remove duplicate patient records, keeping only the first encounter
# This ensures each patient is only represented once in the dataset
df2 = df.drop_duplicates(subset=['patient_nbr'], keep='first')

# Standardization function: Converts data to a standard normal distribution (mean=0, std=1)
def standardize(raw_data):
    return (raw_data - np.mean(raw_data, axis=0)) / np.std(raw_data, axis=0)

# Apply standardization to numeric features
df2[numerics] = standardize(df2[numerics])

# Outlier removal: Keep only data within 3 standard deviations (Z-score method)
df2 = df2[(np.abs(sp.stats.zscore(df2[numerics])) < 3).all(axis=1)]

# Define a colormap for correlation matrix visualization
my_cmap = ListedColormap(sns.light_palette((250, 100, 50), input="husl", n_colors=50).as_hex())

# Compute and visualize correlation matrix after removing unnecessary columns
table = df2.drop(['patient_nbr', 'encounter_id'], axis=1).corr(method='pearson')
table.style.background_gradient(cmap=my_cmap, axis=0)

# Compute absolute correlation values and sort them to analyze feature relationships
c = df2.corr().abs()
s = c.unstack()
so = s.sort_values(ascending=False)
so[38:120]  # Display selected correlation values

# Convert level1_diag1 to categorical for one-hot encoding
df2['level1_diag1'] = df2['level1_diag1'].astype('object')

# Apply one-hot encoding to categorical features
df_pd = pd.get_dummies(df2, columns=['race', 'gender', 'admission_type_id', 'discharge_disposition_id',
                                      'admission_source_id', 'max_glu_serum', 'A1Cresult', 'level1_diag1'], drop_first=True)

# Define categorical and numerical feature lists
non_num_cols = ['race', 'gender', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
                'max_glu_serum', 'A1Cresult', 'level1_diag1']
num_cols = list(set(df._get_numeric_data().columns) - {'readmitted', 'change'})

# Extract newly created categorical columns from one-hot encoding
new_non_num_cols = [j for i in non_num_cols for j in df_pd.columns if i in j]

# Identify interaction features created earlier
interaction_features = [feature for feature in df_pd.columns if '|' in feature]

# Define feature sets for modeling
feature_set_1 = ['age', 'time_in_hospital', 'num_procedures', 'num_medications', 'number_outpatient_log1p', 
                 'number_emergency_log1p', 'number_inpatient_log1p', 'number_diagnoses', 'metformin', 
                 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'glipizide', 'glyburide',
                 'pioglitazone', 'rosiglitazone', 'acarbose', 'tolazamide', 'insulin', 'glyburide-metformin',
                 'race_AfricanAmerican', 'race_Asian', 'race_Caucasian', 'race_Hispanic', 'race_Other', 'gender_1', 
                 'admission_type_id_3', 'admission_type_id_5', 'discharge_disposition_id_2', 'discharge_disposition_id_7', 
                 'discharge_disposition_id_10', 'discharge_disposition_id_18', 'admission_source_id_4',
                 'admission_source_id_7', 'admission_source_id_9', 'max_glu_serum_0', 'max_glu_serum_1', 'A1Cresult_0',
                 'A1Cresult_1', 'num_medications|time_in_hospital', 'num_medications|num_procedures',
                 'time_in_hospital|num_lab_procedures', 'num_medications|num_lab_procedures', 'num_medications|number_diagnoses',
                 'age|number_diagnoses', 'change|num_medications', 'number_diagnoses|time_in_hospital',
                 'num_medications|numchange', 'level1_diag1_1.0', 'level1_diag1_2.0', 'level1_diag1_3.0', 'level1_diag1_4.0',
                 'level1_diag1_5.0','level1_diag1_6.0', 'level1_diag1_7.0', 'level1_diag1_8.0']

feature_set_2 = ['age', 'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'service_utilization_log1p', 
                 'number_diagnoses', 'nummed', 'race_AfricanAmerican', 'race_Asian', 'race_Caucasian', 'race_Hispanic',
                 'race_Other', 'gender_1', 'A1Cresult_0', 'A1Cresult_1', 'admission_type_id_3', 'admission_type_id_4',
                 'admission_type_id_5', 'discharge_disposition_id_2', 'discharge_disposition_id_7',
                 'discharge_disposition_id_10', 'discharge_disposition_id_18', 'admission_source_id_4',
                 'admission_source_id_7', 'admission_source_id_8', 'admission_source_id_9', 'admission_source_id_11',
                 'numchange', 'num_medications|time_in_hospital', 'num_medications|num_procedures',
                 'time_in_hospital|num_lab_procedures', 'num_medications|num_lab_procedures', 'num_medications|number_diagnoses',
                 'age|number_diagnoses', 'change|num_medications', 'number_diagnoses|time_in_hospital',
                 'num_medications|numchange', 'level1_diag1_1.0', 'level1_diag1_2.0', 'level1_diag1_3.0', 'level1_diag1_4.0',
                 'level1_diag1_5.0', 'level1_diag1_6.0', 'level1_diag1_7.0', 'level1_diag1_8.0']

# Prepare training data using Feature Set 1
train_input = df_pd[feature_set_1]
train_output = df_pd['readmitted']

# Split data into training and development sets
X_train, X_dev, Y_train, Y_dev = train_test_split(train_input, train_output, test_size=0.20, random_state=0)

# Initialize logistic regression model with L1 regularization
logreg = LogisticRegression(fit_intercept=True, penalty='l1', solver='liblinear')

# Perform 10-fold cross-validation and print the average score
cv_score = np.mean(cross_val_score(logreg, X_train, Y_train, cv=10))
print(f"Cross Validation Score: {cv_score:.2%}")

# Train logistic regression model
logreg.fit(X_train, Y_train)

# Evaluate the model on the development set
dev_score = logreg.score(X_dev, Y_dev)
print(f"Dev Set score: {dev_score:.2%}")

# Predict the labels for the development set
Y_dev_predict = logreg.predict(X_dev)

# Create a confusion matrix to evaluate predictions
conf_matrix = pd.crosstab(pd.Series(Y_dev, name='Actual'), pd.Series(Y_dev_predict, name='Predict'), margins=True)

# Calculate performance metrics
accuracy = accuracy_score(Y_dev, Y_dev_predict)
precision = precision_score(Y_dev, Y_dev_predict)
recall = recall_score(Y_dev, Y_dev_predict)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=20)
train_input_new, train_output_new = smote.fit_resample(train_input, train_output)

# Convert the balanced dataset back into a DataFrame
train_input_new = pd.DataFrame(train_input_new, columns=train_input.columns)

# Split the balanced data into training and development sets
X_train, X_dev, Y_train, Y_dev = train_test_split(train_input_new, train_output_new, test_size=0.20, random_state=0)

# Retrain logistic regression on balanced data
logreg = LogisticRegression(fit_intercept=True, penalty='l1', solver='liblinear')

# Perform cross-validation
cv_score = np.mean(cross_val_score(logreg, X_train, Y_train, cv=10))
print(f"Cross Validation Score: {cv_score:.2%}")

# Train the model and evaluate
logreg.fit(X_train, Y_train)
dev_score = logreg.score(X_dev, Y_dev)
print(f"Dev Set score: {dev_score:.2%}")

# Predict and evaluate
Y_dev_predict = logreg.predict(X_dev)
conf_matrix = pd.crosstab(pd.Series(Y_dev, name='Actual'), pd.Series(Y_dev_predict, name='Predict'), margins=True)
print(conf_matrix)

# Compute metrics
accuracy = accuracy_score(Y_dev, Y_dev_predict)
precision = precision_score(Y_dev, Y_dev_predict)
recall = recall_score(Y_dev, Y_dev_predict)
auc = roc_auc_score(Y_dev, Y_dev_predict)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"AUC: {auc:.2f}")

# Store results
accuracy_logreg = accuracy
precision_logreg = precision
recall_logreg = recall
auc_logreg = auc

# Perform logistic regression using statsmodels
logit = sm.Logit(Y_train, X_train)
result = logit.fit()

# Print the summary of the logistic regression model
print(result.summary())

# Extract coefficients and p-values
logit_coefs = pd.DataFrame({'Feature': X_train.columns, 'Coefficient': result.params, 'pVal': result.pvalues})

# Filter for significant coefficients (p-value < 0.01)
logit_coefs = logit_coefs[logit_coefs['pVal'] < 0.01]

# Sort by coefficient value
logit_coefs = logit_coefs.sort_values(by='Coefficient', ascending=False)

# Define feature sets
feature_set_1_no_int = ['age', 'time_in_hospital', 'num_procedures', 'num_medications', 'number_outpatient_log1p', 
                 'number_emergency_log1p', 'number_inpatient_log1p', 'number_diagnoses', 'metformin', 
                 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'glipizide', 
                 'glyburide', 'pioglitazone', 'rosiglitazone', 'acarbose', 
                 'tolazamide', 'insulin', 'glyburide-metformin',
                 'race_AfricanAmerican', 'race_Asian', 'race_Caucasian', 
                 'race_Hispanic', 'race_Other', 'gender_1', 
                 'admission_type_id_3', 'admission_type_id_5', 
                 'discharge_disposition_id_2', 'discharge_disposition_id_7', 
                 'discharge_disposition_id_10', 'discharge_disposition_id_18', 
                 'admission_source_id_4', 'admission_source_id_7', 
                 'admission_source_id_9', 'max_glu_serum_0', 
                 'max_glu_serum_1', 'A1Cresult_0', 'A1Cresult_1', 
                 'level1_diag1_1.0',
                 'level1_diag1_2.0',
                 'level1_diag1_3.0',
                 'level1_diag1_4.0',
                 'level1_diag1_5.0',
                 'level1_diag1_6.0',
                 'level1_diag1_7.0',
                 'level1_diag1_8.0']

feature_set_2_no_int = ['age', 'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'service_utilization_log1p', 
'number_diagnoses', 'nummed', 'race_AfricanAmerican', 'race_Asian', 
'race_Caucasian', 'race_Hispanic', 'race_Other', 'gender_1', 
'admission_type_id_3', 'admission_type_id_4', 'admission_type_id_5', 
'discharge_disposition_id_2', 'discharge_disposition_id_7', 
'discharge_disposition_id_10', 'discharge_disposition_id_18', 
'admission_source_id_4', 'admission_source_id_7', 'admission_source_id_8', 
'admission_source_id_9', 'admission_source_id_11', 'numchange', 
'A1Cresult_0', 'A1Cresult_1','level1_diag1_1.0','level1_diag1_2.0','level1_diag1_3.0','level1_diag1_4.0','level1_diag1_5.0',
'level1_diag1_6.0', 'level1_diag1_7.0','level1_diag1_8.0']

# Select relevant features and target variable
train_input = df_pd[feature_set_1_no_int]
train_output = df_pd['readmitted']

# Apply SMOTE for balancing
smt = SMOTE(random_state=20)
train_input_new, train_output_new = smt.fit_resample(train_input, train_output)

# Convert back to DataFrame
train_input_new = pd.DataFrame(train_input_new, columns=train_input.columns)

# Split into train and dev sets
X_train, X_dev, Y_train, Y_dev = train_test_split(train_input_new, train_output_new, test_size=0.20, random_state=0)

# Initialize Decision Tree with optimized hyperparameters
dte = DecisionTreeClassifier(max_depth=28, criterion="entropy", min_samples_split=10)

# Cross-validation score
cv_score = np.mean(cross_val_score(dte, X_train, Y_train, cv=10))
print(f"Cross Validation Score: {cv_score:.2%}")

# Train the model
dte.fit(X_train, Y_train)

# Evaluate on development set
dev_score = dte.score(X_dev, Y_dev)
print(f"Dev Set score: {dev_score:.2%}")

# Make predictions
Y_dev_predict = dte.predict(X_dev)

# Confusion matrix
conf_matrix = pd.crosstab(pd.Series(Y_dev, name='Actual'), pd.Series(Y_dev_predict, name='Predict'), margins=True)

# Compute performance metrics
accuracy = accuracy_score(Y_dev, Y_dev_predict)
precision = precision_score(Y_dev, Y_dev_predict)
recall = recall_score(Y_dev, Y_dev_predict)
auc = roc_auc_score(Y_dev, Y_dev_predict)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"AUC: {auc:.2f}")

# Store results
accuracy_dte = accuracy
precision_dte = precision
recall_dte = recall
auc_dte = auc

# Visualizing decision tree for Question 2 - detailed feature set
plt.figure(figsize=(12, 8))  
tree.plot_tree(dte, 
               feature_names=X_train.columns, 
               max_depth=2, 
               class_names=["No", "Readm"], 
               filled=True, 
               rounded=True)
plt.show()

# Identifying top 10 most important features in the Decision Tree model
feature_imports = dte.feature_importances_
most_imp_features = pd.DataFrame(zip(X_train.columns, feature_imports), columns=["Feature", "Importance"]).nlargest(10, "Importance")

# Plotting feature importance
plt.figure(figsize=(10, 6))
plt.barh(most_imp_features["Feature"], most_imp_features["Importance"], align='center', alpha=0.8)
plt.xlabel('Importance')
plt.title('Most important features - Decision Tree (entropy function, complex model)')
plt.show()

# Training a Decision Tree model with Gini criterion
dtg = DecisionTreeClassifier(max_depth=28, criterion="gini", min_samples_split=10)
print(f"Cross Validation Score: {np.mean(cross_val_score(dtg, X_train, Y_train, cv=10)):.2%}")

dtg.fit(X_train, Y_train)
print(f"Dev Set score: {dtg.score(X_dev, Y_dev):.2%}")

# Making predictions on the development set
Y_dev_predict = dtg.predict(X_dev)

# Confusion matrix
pd.crosstab(pd.Series(Y_dev, name='Actual'), pd.Series(Y_dev_predict, name='Predict'), margins=True)

# Evaluating model performance
accuracy_dtg = accuracy_score(Y_dev, Y_dev_predict)
precision_dtg = precision_score(Y_dev, Y_dev_predict)
recall_dtg = recall_score(Y_dev, Y_dev_predict)
auc_dtg = roc_auc_score(Y_dev, Y_dev_predict)

print(f"Accuracy: {accuracy_dtg:.2f}")
print(f"Precision: {precision_dtg:.2f}")
print(f"Recall: {recall_dtg:.2f}")
print(f"AUC: {auc_dtg:.2f}")

# Applying SMOTE for class balancing
smt = SMOTE(random_state=20)
train_input_new, train_output_new = smt.fit_resample(df_pd[feature_set_1_no_int], df_pd['readmitted'])

# Splitting data into training and development sets
X_train, X_dev, Y_train, Y_dev = train_test_split(train_input_new, train_output_new, test_size=0.20, random_state=0)

# Training a Random Forest model
forrest = RandomForestClassifier(n_estimators=10, max_depth=25, criterion="gini", min_samples_split=10)
print(f"Cross Validation score: {np.mean(cross_val_score(forrest, X_train, Y_train, cv=10)):.2%}")

forrest.fit(X_train, Y_train)
print(f"Dev Set score: {forrest.score(X_dev, Y_dev):.2%}")

# Making predictions and evaluating Random Forest model
Y_dev_predict = forrest.predict(X_dev)
pd.crosstab(pd.Series(Y_dev, name='Actual'), pd.Series(Y_dev_predict, name='Predict'), margins=True)

accuracy_forrestg = accuracy_score(Y_dev, Y_dev_predict)
precision_forrestg = precision_score(Y_dev, Y_dev_predict)
recall_forrestg = recall_score(Y_dev, Y_dev_predict)
auc_forrestg = roc_auc_score(Y_dev, Y_dev_predict)

print(f"Accuracy: {accuracy_forrestg:.2f}")
print(f"Precision: {precision_forrestg:.2f}")
print(f"Recall: {recall_forrestg:.2f}")
print(f"AUC: {auc_forrestg:.2f}")

# Identifying top 10 most important features in the Random Forest model
feature_imports = forrest.feature_importances_
most_imp_features = pd.DataFrame(zip(X_train.columns, feature_imports), columns=["Feature", "Importance"]).nlargest(10, "Importance")

# Plotting feature importance for Random Forest
plt.figure(figsize=(10, 6))
plt.barh(most_imp_features["Feature"], most_imp_features["Importance"], align='center', alpha=0.8)
plt.xlabel('Importance')
plt.title('Most important features - Random Forest (Gini function, complex model)')
plt.show()

# Comparing accuracy of multiple models
models = ['Logistic Regression', 'Decision Tree Gini', 'Decision Tree Entropy', 'Random Forests Gini', 'Random Forests Entropy']
accuracies = [accuracy_logreg, accuracy_dtg, accuracy_dte, accuracy_forrestg, accuracy_forrestg]
precisions = [precision_logreg, precision_dtg, precision_dte, precision_forrestg, precision_forrestg]

plt.figure(figsize=(14, 5))
bar_width = 0.15
indices = np.arange(len(models))

# Plotting accuracy
plt.bar(indices, accuracies, width=bar_width, alpha=0.7, color='red', label='Accuracy')

# Plotting precision (shifted bars)
plt.bar(indices + bar_width, precisions, width=bar_width, alpha=0.7, color='blue', label='Precision')

plt.xticks(indices, models, rotation=15)
plt.legend()
plt.title('Model Performance Comparison')
plt.show()

ax = plt.subplot(111)

models = ['Logistic Regression', 'Decision Tree Gini', 'Decision Tree Entropy', 'Random Forests Gini', 'Random Forests Entropy' ]
values = [recall_logreg, recall_dtg, recall_dte, recall_forrestg, recall_forrestg]
model = np.arange(len(models))

plt.bar(model+0.3, values, align='center', width = 0.15, alpha=0.7, color = 'green', label = 'recall')
plt.xticks(model, models)

# ---- Visualization for performance metrics ----

# Set up the plot for recall values (comparison between models)
ax = plt.subplot(111)

# List of models and their respective recall values (make sure these variables are already calculated)
models = ['Logistic Regression', 'Decision Tree Gini', 'Decision Tree Entropy', 'Random Forests Gini', 'Random Forests Entropy']
values = [recall_logreg, recall_dtg, recall_dte, recall_forrestg, recall_forrestg]
model = np.arange(len(models))  # Create an index for the models

# Plot recall values
plt.bar(model + 0.3, values, align='center', width=0.15, alpha=0.7, color='green', label='Recall')
plt.xticks(model, models)

# ---- Visualization for AUC values ----

# Plot AUC values on the same axis
ax = plt.subplot(111)
values = [auc_logreg, auc_dtg, auc_dte, auc_forrestg, auc_forrestg]

# Plot AUC values
plt.bar(model + 0.45, values, align='center', width=0.15, alpha=0.7, color='orange', label='AUC')
plt.xticks(model, models)

# Label the plot and show
plt.ylabel('Performance Metrics for Different Models')
plt.title('Model Performance Comparison')
ax.spines['right'].set_visible(False) 
ax.spines['top'].set_visible(False)
ax.legend()
plt.show()

# ---- Feature Importance from Random Forest ----

# Calculate and sort the feature importances from Random Forest (for model interpretation)
impforrest = [(x[1], x[0]) for x in sorted(zip(list(forrest.feature_importances_), list(train_input.columns)), reverse=True)]
print("Random Forest Feature Importance in decreasing order")

# Display feature importances as a dataframe
impforrest = pd.DataFrame(impforrest, columns=["Feature", "Importance"])
print(impforrest)

# ---- Merge Logistic Regression Coefficients and Random Forest Feature Importances ----

# Merge logistic regression coefficients with Random Forest feature importance to compare both
coef_imp = logit_coefs.merge(impforrest, how='inner', on=['Feature'])
print(coef_imp)

# ---- Coefficients vs Feature Importance Visualization ----

# Plot the absolute values of logistic regression coefficients vs Random Forest feature importances
plt.figure(figsize=(18, 10))
ax = plt.subplot(111)
x = coef_imp.Coefficient.abs()
y = coef_imp.Importance

# Scatter plot of coefficients vs feature importance
plt.scatter(x, y, color='blue')

# Remove the top and right axis for aesthetics
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Label the axes and the plot
plt.xlabel("Absolute of Coefficient (Logistic Regression)")
plt.ylabel("Mean Feature Importance (Random Forest)")
plt.title("Coefficients vs Feature Importance - Complex Model")

# Annotate points where coefficient or importance is large
for label, m, n in zip(coef_imp.Feature, coef_imp.Coefficient.abs(), coef_imp.Importance):
    if (m > 0.4) or (n > 0.03):
        plt.annotate(label, xy=(m, n), xytext=(25, 8), textcoords='offset points', ha='right', va='bottom')

plt.show()

# ---- Data Preprocessing and SMOTE ----

# Prepare dataset for simple model (assumed to be defined earlier)
train_input = df_pd[feature_set_2]
train_output = df_pd['readmitted']

# Display the original dataset shape (before SMOTE)
print('Original dataset shape {}'.format(Counter(train_output)))

# Apply SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset
smote = SMOTE(random_state=20)
train_input_new, train_output_new = smote.fit_resample(train_input, train_output)

# Display the new dataset shape after SMOTE
print('New dataset shape {}'.format(Counter(train_output_new)))

# Convert the balanced dataset to DataFrame
train_input_new = pd.DataFrame(train_input_new, columns=list(train_input.columns))

# ---- Train-Test Split for Logistic Regression ----

# Split the balanced dataset into training and development sets
X_train, X_dev, Y_train, Y_dev = train_test_split(train_input_new, train_output_new, test_size=0.20, random_state=0)

# Logistic Regression model fitting
logreg = LogisticRegression(fit_intercept=True)

# Cross-validation score
print("Cross Validation Score: {:.2%}".format(np.mean(cross_val_score(logreg, X_train, Y_train, cv=10))))

# Train the logistic regression model
logreg.fit(X_train, Y_train)

# Dev set accuracy
print("Dev Set score: {:.2%}".format(logreg.score(X_dev, Y_dev)))

# ---- Logistic Regression Summary using Statsmodels ----

# Logit model for further analysis (statsmodels)
logit2 = sm.Logit(Y_train, X_train)
result2 = logit2.fit()

# Modify chi-square probability calculation for better interpretation
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

# Print result summary (coefficients and p-values)
print(result2.summary())

# ---- Extracting Logistic Regression Coefficients and P-values ----

# Extract coefficients and p-values from logistic regression result
logit2_coefs = pd.DataFrame(result2.params)
logit2_coefs.reset_index(level=0, inplace=True)
logit2_coefs.columns = ["Feature", "Coefficient"]

logit2_pvals = pd.DataFrame(result2.pvalues)
logit2_pvals.reset_index(level=0, inplace=True)
logit2_pvals.columns = ["Feature", "pVal"]

# Merge coefficients and p-values
logit2_coefs = logit2_coefs.merge(logit2_pvals, how="inner", on=["Feature"])

# Filter features with p-value < 0.01
logit2_coefs = logit2_coefs[logit2_coefs.pVal < 0.01]
print(logit2_coefs)

# ---- Predicting and Evaluating Model Performance ----

# Predict the development set labels using logistic regression
Y_dev_predict = result2.predict(X_dev, linear=False)
Y_dev_predict = (Y_dev_predict > 0.5).astype(int)

# Display the confusion matrix
print(pd.crosstab(pd.Series(Y_dev, name='Actual'), pd.Series(Y_dev_predict, name='Predict'), margins=True))

# ---- Performance Metrics ----

# Calculate and display performance metrics
print("Accuracy is {0:.2f}".format(accuracy_score(Y_dev, Y_dev_predict)))
print("Precision is {0:.2f}".format(precision_score(Y_dev, Y_dev_predict)))
print("Recall is {0:.2f}".format(recall_score(Y_dev, Y_dev_predict)))
print("AUC is {0:.2f}".format(roc_auc_score(Y_dev, Y_dev_predict)))

# Store performance metrics for later use in the plots
accuracy_logreg = accuracy_score(Y_dev, Y_dev_predict)
precision_logreg = precision_score(Y_dev, Y_dev_predict)
recall_logreg = recall_score(Y_dev, Y_dev_predict)
auc_logreg = roc_auc_score(Y_dev, Y_dev_predict)

# # ## Decision Tree for Q2 - Simple model
# --- Data Preprocessing: Resampling and Splitting ---
train_input = df_pd[feature_set_2_no_int]  # Selecting the feature set without interaction terms
train_output = df_pd['readmitted']  # Target variable

# Checking class distribution
print('Original dataset shape {}'.format(Counter(train_output)))
smote = SMOTE(random_state=20)  # Applying SMOTE for balancing the class distribution
train_input_new, train_output_new = smote.fit_resample(train_input, train_output)
print('New dataset shape {}'.format(Counter(train_output_new)))

train_input_new = pd.DataFrame(train_input_new, columns=list(train_input.columns))  # Converting to DataFrame
X_train, X_dev, Y_train, Y_dev = train_test_split(train_input_new, train_output_new, test_size=0.20, random_state=0)  # Splitting into train and dev sets


# --- Decision Tree - Entropy (Simple Model) ---
dte = DecisionTreeClassifier(max_depth=28, criterion="entropy", min_samples_split=10)
print("Cross Validation score: {:.2%}".format(np.mean(cross_val_score(dte, X_train, Y_train, cv=10))))
dte.fit(X_train, Y_train)
print("Dev Set score: {:.2%}".format(dte.score(X_dev, Y_dev)))

Y_dev_predict = dte.predict(X_dev)

# Display confusion matrix
pd.crosstab(pd.Series(Y_dev, name='Actual'), pd.Series(Y_dev_predict, name='Predict'), margins=True)

# Performance metrics
print("Accuracy: {:.2f}".format(accuracy_score(Y_dev, Y_dev_predict)))
print("Precision: {:.2f}".format(precision_score(Y_dev, Y_dev_predict)))
print("Recall: {:.2f}".format(recall_score(Y_dev, Y_dev_predict)))
print("AUC: {:.2f}".format(roc_auc_score(Y_dev, Y_dev_predict)))

# Store metrics for comparison
accuracy_dte = accuracy_score(Y_dev, Y_dev_predict)
precision_dte = precision_score(Y_dev, Y_dev_predict)
recall_dte = recall_score(Y_dev, Y_dev_predict)
auc_dte = roc_auc_score(Y_dev, Y_dev_predict)

# Visualization of decision tree
plt.figure(figsize=(12, 8))
tree.plot_tree(dte, feature_names=X_train.columns, max_depth=2, class_names=["No", "Readm"], filled=True, rounded=True)

plt.show()

# Feature importance visualization
feature_names = X_train.columns
feature_imports = dte.feature_importances_
most_imp_features = pd.DataFrame([f for f in zip(feature_names, feature_imports)], columns=["Feature", "Importance"]).nlargest(10, "Importance")
most_imp_features.sort_values(by="Importance", inplace=True)
plt.figure(figsize=(10, 6))
plt.barh(range(len(most_imp_features)), most_imp_features.Importance, align='center', alpha=0.8)
plt.yticks(range(len(most_imp_features)), most_imp_features.Feature, fontsize=14)
plt.xlabel('Importance')
plt.title('Most important features - Decision Tree (entropy)')
plt.show()


# --- Decision Tree - Gini (Simple Model) ---
dtg = DecisionTreeClassifier(max_depth=28, criterion="gini", min_samples_split=10)
print("Cross Validation Score: {:.2%}".format(np.mean(cross_val_score(dtg, X_train, Y_train, cv=10))))
dtg.fit(X_train, Y_train)
print("Dev Set score: {:.2%}".format(dtg.score(X_dev, Y_dev)))

Y_dev_predict = dtg.predict(X_dev)

# Performance metrics
print("Accuracy: {:.2f}".format(accuracy_score(Y_dev, Y_dev_predict)))
print("Precision: {:.2f}".format(precision_score(Y_dev, Y_dev_predict)))
print("Recall: {:.2f}".format(recall_score(Y_dev, Y_dev_predict)))
print("AUC: {:.2f}".format(roc_auc_score(Y_dev, Y_dev_predict)))

# Store metrics for comparison
accuracy_dtg = accuracy_score(Y_dev, Y_dev_predict)
precision_dtg = precision_score(Y_dev, Y_dev_predict)
recall_dtg = recall_score(Y_dev, Y_dev_predict)
auc_dtg = roc_auc_score(Y_dev, Y_dev_predict)


# --- Random Forest - Entropy (Simple Model) ---
forrest = RandomForestClassifier(n_estimators=10, max_depth=25, criterion="entropy", min_samples_split=10)
print("Cross Validation score: {:.2%}".format(np.mean(cross_val_score(forrest, X_train, Y_train, cv=10))))
forrest.fit(X_train, Y_train)
print("Dev Set score: {:.2%}".format(forrest.score(X_dev, Y_dev)))

Y_dev_predict = forrest.predict(X_dev)

# Performance metrics
print("Accuracy: {:.2f}".format(accuracy_score(Y_dev, Y_dev_predict)))
print("Precision: {:.2f}".format(precision_score(Y_dev, Y_dev_predict)))
print("Recall: {:.2f}".format(recall_score(Y_dev, Y_dev_predict)))
print("AUC: {:.2f}".format(roc_auc_score(Y_dev, Y_dev_predict)))

# Store metrics for comparison
accuracy_forreste = accuracy_score(Y_dev, Y_dev_predict)
precision_forreste = precision_score(Y_dev, Y_dev_predict)
recall_forreste = recall_score(Y_dev, Y_dev_predict)
auc_forreste = roc_auc_score(Y_dev, Y_dev_predict)

# Feature importance visualization
feature_names = X_train.columns
feature_imports = forrest.feature_importances_
most_imp_features = pd.DataFrame([f for f in zip(feature_names, feature_imports)], columns=["Feature", "Importance"]).nlargest(10, "Importance")
most_imp_features.sort_values(by="Importance", inplace=True)
plt.figure(figsize=(10, 6))
plt.barh(range(len(most_imp_features)), most_imp_features.Importance, align='center', alpha=0.8)
plt.yticks(range(len(most_imp_features)), most_imp_features.Feature, fontsize=14)
plt.xlabel('Importance')
plt.title('Most important features - Random Forest (entropy)')
plt.show()


# --- Random Forest - Gini (Simple Model) ---
forrest = RandomForestClassifier(n_estimators=10, max_depth=27, criterion="gini", min_samples_split=10)
print("Cross Validation score: {:.2%}".format(np.mean(cross_val_score(forrest, X_train, Y_train, cv=10))))
forrest.fit(X_train, Y_train)
print("Dev Set score: {:.2%}".format(forrest.score(X_dev, Y_dev)))

Y_dev_predict = forrest.predict(X_dev)

# Performance metrics
print("Accuracy: {:.2f}".format(accuracy_score(Y_dev, Y_dev_predict)))
print("Precision: {:.2f}".format(precision_score(Y_dev, Y_dev_predict)))
print("Recall: {:.2f}".format(recall_score(Y_dev, Y_dev_predict)))
print("AUC: {:.2f}".format(roc_auc_score(Y_dev, Y_dev_predict)))

# Store metrics for comparison
accuracy_forrestg = accuracy_score(Y_dev, Y_dev_predict)
precision_forrestg = precision_score(Y_dev, Y_dev_predict)
recall_forrestg = recall_score(Y_dev, Y_dev_predict)
auc_forrestg = roc_auc_score(Y_dev, Y_dev_predict)

# --- Feature Importance Visualization for Random Forest ---
feature_names = X_train.columns
feature_imports = forrest.feature_importances_

# Create dataframe of feature importance and get top 10 features
most_imp_features = pd.DataFrame(list(zip(feature_names, feature_imports)), columns=["Feature", "Importance"]).nlargest(10, "Importance")
most_imp_features.sort_values(by="Importance", inplace=True)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(range(len(most_imp_features)), most_imp_features.Importance, align='center', alpha=0.8)
plt.yticks(range(len(most_imp_features)), most_imp_features.Feature, fontsize=14)
plt.xlabel('Importance')
plt.title('Most important features - Random Forest (Gini function, Simple model)')
plt.show()


# --- Model Performance Visualization ---
# Model names and corresponding metrics
models = ['Logistic Regression', 'Decision Tree Gini', 'Decision Tree Entropy', 'Random Forests Gini', 'Random Forests Entropy']
accuracy_values = [accuracy_logreg, accuracy_dtg, accuracy_dte, accuracy_forrestg, accuracy_forreste]
precision_values = [precision_logreg, precision_dtg, precision_dte, precision_forrestg, precision_forreste]
recall_values = [recall_logreg, recall_dtg, recall_dte, recall_forrestg, recall_forreste]
auc_values = [auc_logreg, auc_dtg, auc_dte, auc_forrestg, auc_forreste]

# Create bar plots for each metric
fig, ax = plt.subplots(figsize=(14, 5))

# Accuracy
ax.bar(np.arange(len(models)), accuracy_values, align='center', width=0.15, alpha=0.7, color='red', label='Accuracy')

# Precision
ax.bar(np.arange(len(models)) + 0.15, precision_values, align='center', width=0.15, alpha=0.7, color='blue', label='Precision')

# Recall
ax.bar(np.arange(len(models)) + 0.3, recall_values, align='center', width=0.15, alpha=0.7, color='green', label='Recall')

# AUC
ax.bar(np.arange(len(models)) + 0.45, auc_values, align='center', width=0.15, alpha=0.7, color='orange', label='AUC')

# Customize the plot
ax.set_xticks(np.arange(len(models)) + 0.225)  # Set x-ticks in the middle of the grouped bars
ax.set_xticklabels(models)
ax.set_ylabel('Performance Metrics for Different Models')
ax.set_title('Model Comparison - Performance Metrics')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.legend()

plt.show()