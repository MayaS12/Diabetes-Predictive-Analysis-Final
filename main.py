# Import libraries needed
import pandas as pd
import numpy as np
from statistics import mode
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load original data into dataframe and check shape
df_ori = pd.read_csv("diabetic_data.csv")
print(df_ori.shape)

# Display the first few rows of the dataframe
print(df_ori.head())

# Make a copy of the dataframe for preprocessing
df = df_ori.copy(deep=True)

# Calculate missing values
missingvalues = pd.DataFrame(df.isnull().sum(axis=0), columns=['number_of_missing_values'])
missingvalues['feature'] = missingvalues.index
missingvalues = missingvalues[['feature', 'number_of_missing_values']].reset_index(drop=True)
print(missingvalues[missingvalues.number_of_missing_values > 0])

# Create service utilization as a new feature
df['service_utilization'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']

# Re-encoding admission type, discharge type and admission source into fewer categories
df['admission_type_id'] = df['admission_type_id'].replace(2, 1)
df['admission_type_id'] = df['admission_type_id'].replace(7, 1)
df['admission_type_id'] = df['admission_type_id'].replace(6, 5)
df['admission_type_id'] = df['admission_type_id'].replace(8, 5)

df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(6, 1)
df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(8, 1)
df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(9, 1)
df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(13, 1)
df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(3, 2)
df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(4, 2)
df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(5, 2)
df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(14, 2)
df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(22, 2)
df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(23, 2)
df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(24, 2)
df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(12, 10)
df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(15, 10)
df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(16, 10)
df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(17, 10)
df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(25, 18)
df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(26, 18)

df['admission_source_id'] = df['admission_source_id'].replace(2, 1)
df['admission_source_id'] = df['admission_source_id'].replace(3, 1)
df['admission_source_id'] = df['admission_source_id'].replace(5, 4)
df['admission_source_id'] = df['admission_source_id'].replace(6, 4)
df['admission_source_id'] = df['admission_source_id'].replace(10, 4)
df['admission_source_id'] = df['admission_source_id'].replace(22, 4)
df['admission_source_id'] = df['admission_source_id'].replace(25, 4)
df['admission_source_id'] = df['admission_source_id'].replace(15, 9)
df['admission_source_id'] = df['admission_source_id'].replace(17, 9)
df['admission_source_id'] = df['admission_source_id'].replace(20, 9)
df['admission_source_id'] = df['admission_source_id'].replace(21, 9)
df['admission_source_id'] = df['admission_source_id'].replace(13, 11)
df['admission_source_id'] = df['admission_source_id'].replace(14, 11)


# Count missing values represented by '?' in categorical columns
for col in df.columns:
    if df[col].dtype == object:
        print(col, df[col][df[col] == '?'].count())
print('gender', df['gender'][df['gender'] == 'Unknown/Invalid'].count())

# Encode the target variable 'readmitted' to numerical values
df['readmitted'] = df['readmitted'].replace('>30', 2)  # More than 30-day readmission -> 2
df['readmitted'] = df['readmitted'].replace('<30', 1)  # Within 30-day readmission -> 1
df['readmitted'] = df['readmitted'].replace('NO', 0)   # No readmission -> 0

# Print counts of each category in 'readmitted'
print('>30 readmissions', df['readmitted'][df['readmitted'] == 2].count())
print('<30 readmissions', df['readmitted'][df['readmitted'] == 1].count())
print('no readmissions', df['readmitted'][df['readmitted'] == 0].count())

# Identify and remove rows where all three diagnosis fields are missing ('?')
drop_Idx = set(df[(df['diag_1'] == '?') & (df['diag_2'] == '?') & (df['diag_3'] == '?')].index)

# Remove patients who were discharged to hospice (discharge_disposition_id == 11)
drop_Idx = drop_Idx.union(set(df[df['discharge_disposition_id'] == 11].index))

# Remove rows where gender is 'Unknown/Invalid'
drop_Idx = drop_Idx.union(set(df['gender'][df['gender'] == 'Unknown/Invalid'].index))

# Keep only the remaining rows
new_Idx = list(set(df.index) - set(drop_Idx))
df = df.iloc[new_Idx]

# Drop irrelevant or redundant columns
df = df.drop(['weight', 'payer_code', 'medical_specialty'], axis=1)

# Drop specific medication columns that are mostly missing or not informative
df = df.drop(['citoglipton', 'examide'], axis=1)

# Convert categorical variable 'change' (whether medication was changed) to binary
df['change'] = df['change'].replace('Ch', 1)  # Change -> 1
df['change'] = df['change'].replace('No', 0)  # No change -> 0

# Convert gender to binary (Male -> 1, Female -> 0)
df['gender'] = df['gender'].replace('Male', 1)
df['gender'] = df['gender'].replace('Female', 0)

# Convert diabetes medication usage to binary (Yes -> 1, No -> 0)
df['diabetesMed'] = df['diabetesMed'].replace('Yes', 1)
df['diabetesMed'] = df['diabetesMed'].replace('No', 0)

# Encode 'age' as ordinal categories (10-year bins)
for i in range(0, 10):
    df['age'] = df['age'].replace('[' + str(10 * i) + '-' + str(10 * (i + 1)) + ')', i + 1)

# List of diabetes-related medications
keys = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'glipizide', 'glyburide',
        'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'insulin', 'glyburide-metformin', 'tolazamide',
        'metformin-pioglitazone', 'metformin-rosiglitazone', 'glimepiride-pioglitazone',
        'glipizide-metformin', 'troglitazone', 'tolbutamide', 'acetohexamide']

# Create temporary binary columns indicating medication changes
for col in keys:
    colname = str(col) + 'temp'
    df[colname] = df[col].apply(lambda x: 0 if (x == 'No' or x == 'Steady') else 1)

# Create a feature 'numchange' to count how many medications were changed
df['numchange'] = 0
for col in keys:
    colname = str(col) + 'temp'
    df['numchange'] = df['numchange'] + df[colname]
    del df[colname]  # Remove temporary columns

# Print distribution of 'numchange'
print(df['numchange'].value_counts())

# Convert medication usage to binary (Steady, Up, and Down -> 1, No -> 0)
for col in keys:
    df[col] = df[col].replace('No', 0)
    df[col] = df[col].replace('Steady', 1)
    df[col] = df[col].replace('Up', 1)
    df[col] = df[col].replace('Down', 1)

# Encode 'A1Cresult' into numeric categories
df['A1Cresult'] = df['A1Cresult'].replace('>7', 1)    # Above 7 -> 1
df['A1Cresult'] = df['A1Cresult'].replace('>8', 1)    # Above 8 -> 1
df['A1Cresult'] = df['A1Cresult'].replace('Norm', 0)  # Normal -> 0
df['A1Cresult'] = df['A1Cresult'].replace('None', -99)  # Missing -> -99

# Encode 'max_glu_serum' (glucose test result)
df['max_glu_serum'] = df['max_glu_serum'].replace('>200', 1)  # Above 200 -> 1
df['max_glu_serum'] = df['max_glu_serum'].replace('>300', 1)  # Above 300 -> 1
df['max_glu_serum'] = df['max_glu_serum'].replace('Norm', 0)  # Normal -> 0
df['max_glu_serum'] = df['max_glu_serum'].replace('None', -99)  # Missing -> -99

# Create a new feature 'nummed' to count the number of medications prescribed
df['nummed'] = 0
for col in keys:
    df['nummed'] = df['nummed'] + df[col]

# Print distribution of 'nummed'
print(df['nummed'].value_counts())

# Create new columns for diagnosis categories
df['level1_diag1'] = df['diag_1']
df['level2_diag1'] = df['diag_1']
df['level1_diag2'] = df['diag_2']
df['level2_diag2'] = df['diag_2']
df['level1_diag3'] = df['diag_3']
df['level2_diag3'] = df['diag_3']

# Convert diagnosis codes starting with 'V' or 'E' to 0 (they represent external causes or supplementary factors)
df.loc[df['diag_1'].str.contains('V'), ['level1_diag1', 'level2_diag1']] = 0
df.loc[df['diag_1'].str.contains('E'), ['level1_diag1', 'level2_diag1']] = 0
df.loc[df['diag_2'].str.contains('V'), ['level1_diag2', 'level2_diag2']] = 0
df.loc[df['diag_2'].str.contains('E'), ['level1_diag2', 'level2_diag2']] = 0
df.loc[df['diag_3'].str.contains('V'), ['level1_diag3', 'level2_diag3']] = 0
df.loc[df['diag_3'].str.contains('E'), ['level1_diag3', 'level2_diag3']] = 0

# Replace missing diagnosis codes ('?') with -1
df['level1_diag1'] = df['level1_diag1'].replace('?', -1)
df['level2_diag1'] = df['level2_diag1'].replace('?', -1)
df['level1_diag2'] = df['level1_diag2'].replace('?', -1)
df['level2_diag2'] = df['level2_diag2'].replace('?', -1)
df['level1_diag3'] = df['level1_diag3'].replace('?', -1)
df['level2_diag3'] = df['level2_diag3'].replace('?', -1)

# Convert diagnosis columns to float for numerical processing
df['level1_diag1'] = df['level1_diag1'].astype(float)
df['level2_diag1'] = df['level2_diag1'].astype(float)
df['level1_diag2'] = df['level1_diag2'].astype(float)
df['level2_diag2'] = df['level2_diag2'].astype(float)
df['level1_diag3'] = df['level1_diag3'].astype(float)
df['level2_diag3'] = df['level2_diag3'].astype(float)

# Iterate through each row in the DataFrame to categorize diagnoses into broader groups
for index, row in df.iterrows():
    # Categorizing 'level1_diag1' based on ICD-9 codes
    if (row['level1_diag1'] >= 390 and row['level1_diag1'] < 460) or (np.floor(row['level1_diag1']) == 785):
        df.loc[index, 'level1_diag1'] = 1  # Circulatory diseases
    elif (row['level1_diag1'] >= 460 and row['level1_diag1'] < 520) or (np.floor(row['level1_diag1']) == 786):
        df.loc[index, 'level1_diag1'] = 2  # Respiratory diseases
    elif (row['level1_diag1'] >= 520 and row['level1_diag1'] < 580) or (np.floor(row['level1_diag1']) == 787):
        df.loc[index, 'level1_diag1'] = 3  # Digestive diseases
    elif (np.floor(row['level1_diag1']) == 250):
        df.loc[index, 'level1_diag1'] = 4  # Diabetes
    elif (row['level1_diag1'] >= 800 and row['level1_diag1'] < 1000):
        df.loc[index, 'level1_diag1'] = 5  # Injury and poisoning
    elif (row['level1_diag1'] >= 710 and row['level1_diag1'] < 740):
        df.loc[index, 'level1_diag1'] = 6  # Musculoskeletal diseases
    elif (row['level1_diag1'] >= 580 and row['level1_diag1'] < 630) or (np.floor(row['level1_diag1']) == 788):
        df.loc[index, 'level1_diag1'] = 7  # Genitourinary diseases
    elif (row['level1_diag1'] >= 140 and row['level1_diag1'] < 240):
        df.loc[index, 'level1_diag1'] = 8  # Neoplasms (cancers)
    else:
        df.loc[index, 'level1_diag1'] = 0  # Other conditions or unspecified

    # Categorizing 'level1_diag2' using the same approach as 'level1_diag1'
    if (row['level1_diag2'] >= 390 and row['level1_diag2'] < 460) or (np.floor(row['level1_diag2']) == 785):
        df.loc[index, 'level1_diag2'] = 1
    elif (row['level1_diag2'] >= 460 and row['level1_diag2'] < 520) or (np.floor(row['level1_diag2']) == 786):
        df.loc[index, 'level1_diag2'] = 2
    elif (row['level1_diag2'] >= 520 and row['level1_diag2'] < 580) or (np.floor(row['level1_diag2']) == 787):
        df.loc[index, 'level1_diag2'] = 3
    elif (np.floor(row['level1_diag2']) == 250):
        df.loc[index, 'level1_diag2'] = 4
    elif (row['level1_diag2'] >= 800 and row['level1_diag2'] < 1000):
        df.loc[index, 'level1_diag2'] = 5
    elif (row['level1_diag2'] >= 710 and row['level1_diag2'] < 740):
        df.loc[index, 'level1_diag2'] = 6
    elif (row['level1_diag2'] >= 580 and row['level1_diag2'] < 630) or (np.floor(row['level1_diag2']) == 788):
        df.loc[index, 'level1_diag2'] = 7
    elif (row['level1_diag2'] >= 140 and row['level1_diag2'] < 240):
        df.loc[index, 'level1_diag2'] = 8
    else:
        df.loc[index, 'level1_diag2'] = 0

    # Categorizing 'level1_diag3' using the same approach as 'level1_diag1'
    if (row['level1_diag3'] >= 390 and row['level1_diag3'] < 460) or (np.floor(row['level1_diag3']) == 785):
        df.loc[index, 'level1_diag3'] = 1
    elif (row['level1_diag3'] >= 460 and row['level1_diag3'] < 520) or (np.floor(row['level1_diag3']) == 786):
        df.loc[index, 'level1_diag3'] = 2
    elif (row['level1_diag3'] >= 520 and row['level1_diag3'] < 580) or (np.floor(row['level1_diag3']) == 787):
        df.loc[index, 'level1_diag3'] = 3
    elif (np.floor(row['level1_diag3']) == 250):
        df.loc[index, 'level1_diag3'] = 4
    elif (row['level1_diag3'] >= 800 and row['level1_diag3'] < 1000):
        df.loc[index, 'level1_diag3'] = 5
    elif (row['level1_diag3'] >= 710 and row['level1_diag3'] < 740):
        df.loc[index, 'level1_diag3'] = 6
    elif (row['level1_diag3'] >= 580 and row['level1_diag3'] < 630) or (np.floor(row['level1_diag3']) == 788):
        df.loc[index, 'level1_diag3'] = 7
    elif (row['level1_diag3'] >= 140 and row['level1_diag3'] < 240):
        df.loc[index, 'level1_diag3'] = 8
    else:
        df.loc[index, 'level1_diag3'] = 0

for index, row in df.iterrows():
    # Mapping values in 'level2_diag1' column to category numbers based on predefined conditions
    if (row['level2_diag1'] >= 390 and row['level2_diag1'] < 399):
        df.loc[index, 'level2_diag1'] = 1
    elif (row['level2_diag1'] >= 401 and row['level2_diag1'] < 415):
        df.loc[index, 'level2_diag1'] = 2
    elif (row['level2_diag1'] >= 415 and row['level2_diag1'] < 460):
        df.loc[index, 'level2_diag1'] = 3
    elif (np.floor(row['level2_diag1']) == 785):
        df.loc[index, 'level2_diag1'] = 4
    elif (row['level2_diag1'] >= 460 and row['level2_diag1'] < 489):
        df.loc[index, 'level2_diag1'] = 5
    elif (row['level2_diag1'] >= 490 and row['level2_diag1'] < 497):
        df.loc[index, 'level2_diag1'] = 6
    elif (row['level2_diag1'] >= 500 and row['level2_diag1'] < 520):
        df.loc[index, 'level2_diag1'] = 7
    elif (np.floor(row['level2_diag1']) == 786):
        df.loc[index, 'level2_diag1'] = 8
    elif (row['level2_diag1'] >= 520 and row['level2_diag1'] < 530):
        df.loc[index, 'level2_diag1'] = 9
    elif (row['level2_diag1'] >= 530 and row['level2_diag1'] < 544):
        df.loc[index, 'level2_diag1'] = 10
    elif (row['level2_diag1'] >= 550 and row['level2_diag1'] < 554):
        df.loc[index, 'level2_diag1'] = 11
    elif (row['level2_diag1'] >= 555 and row['level2_diag1'] < 580):
        df.loc[index, 'level2_diag1'] = 12
    elif (np.floor(row['level2_diag1']) == 787):
        df.loc[index, 'level2_diag1'] = 13
    elif (np.floor(row['level2_diag1']) == 250):
        df.loc[index, 'level2_diag1'] = 14
    elif (row['level2_diag1'] >= 800 and row['level2_diag1'] < 1000):
        df.loc[index, 'level2_diag1'] = 15
    elif (row['level2_diag1'] >= 710 and row['level2_diag1'] < 740):
        df.loc[index, 'level2_diag1'] = 16
    elif (row['level2_diag1'] >= 580 and row['level2_diag1'] < 630):
        df.loc[index, 'level2_diag1'] = 17
    elif (np.floor(row['level2_diag1']) == 788):
        df.loc[index, 'level2_diag1'] = 18
    elif (row['level2_diag1'] >= 140 and row['level2_diag1'] < 240):
        df.loc[index, 'level2_diag1'] = 19
    elif row['level2_diag1'] >= 240 and row['level2_diag1'] < 280 and (np.floor(row['level2_diag1']) != 250):
        df.loc[index, 'level2_diag1'] = 20
    elif (row['level2_diag1'] >= 680 and row['level2_diag1'] < 710) or (np.floor(row['level2_diag1']) == 782):
        df.loc[index, 'level2_diag1'] = 21
    elif (row['level2_diag1'] >= 290 and row['level2_diag1'] < 320):
        df.loc[index, 'level2_diag1'] = 22
    else:
        df.loc[index, 'level2_diag1'] = 0

    # Repeat the same mapping logic for 'level2_diag2'
    if (row['level2_diag2'] >= 390 and row['level2_diag2'] < 399):
        df.loc[index, 'level2_diag2'] = 1
    elif (row['level2_diag2'] >= 401 and row['level2_diag2'] < 415):
        df.loc[index, 'level2_diag2'] = 2
    elif (row['level2_diag2'] >= 415 and row['level2_diag2'] < 460):
        df.loc[index, 'level2_diag2'] = 3
    elif (np.floor(row['level2_diag2']) == 785):
        df.loc[index, 'level2_diag2'] = 4
    elif (row['level2_diag2'] >= 460 and row['level2_diag2'] < 489):
        df.loc[index, 'level2_diag2'] = 5
    elif (row['level2_diag2'] >= 490 and row['level2_diag2'] < 497):
        df.loc[index, 'level2_diag2'] = 6
    elif (row['level2_diag2'] >= 500 and row['level2_diag2'] < 520):
        df.loc[index, 'level2_diag2'] = 7
    elif (np.floor(row['level2_diag2']) == 786):
        df.loc[index, 'level2_diag2'] = 8
    elif (row['level2_diag2'] >= 520 and row['level2_diag2'] < 530):
        df.loc[index, 'level2_diag2'] = 9
    elif (row['level2_diag2'] >= 530 and row['level2_diag2'] < 544):
        df.loc[index, 'level2_diag2'] = 10
    elif (row['level2_diag2'] >= 550 and row['level2_diag2'] < 554):
        df.loc[index, 'level2_diag2'] = 11
    elif (row['level2_diag2'] >= 555 and row['level2_diag2'] < 580):
        df.loc[index, 'level2_diag2'] = 12
    elif (np.floor(row['level2_diag2']) == 787):
        df.loc[index, 'level2_diag2'] = 13
    elif (np.floor(row['level2_diag2']) == 250):
        df.loc[index, 'level2_diag2'] = 14
    elif (row['level2_diag2'] >= 800 and row['level2_diag2'] < 1000):
        df.loc[index, 'level2_diag2'] = 15
    elif (row['level2_diag2'] >= 710 and row['level2_diag2'] < 740):
        df.loc[index, 'level2_diag2'] = 16
    elif (row['level2_diag2'] >= 580 and row['level2_diag2'] < 630):
        df.loc[index, 'level2_diag2'] = 17
    elif (np.floor(row['level2_diag2']) == 788):
        df.loc[index, 'level2_diag2'] = 18
    elif (row['level2_diag2'] >= 140 and row['level2_diag2'] < 240):
        df.loc[index, 'level2_diag2'] = 19
    elif row['level2_diag2'] >= 240 and row['level2_diag2'] < 280 and (np.floor(row['level2_diag2']) != 250):
        df.loc[index, 'level2_diag2'] = 20
    elif (row['level2_diag2'] >= 680 and row['level2_diag2'] < 710) or (np.floor(row['level2_diag2']) == 782):
        df.loc[index, 'level2_diag2'] = 21
    elif (row['level2_diag2'] >= 290 and row['level2_diag2'] < 320):
        df.loc[index, 'level2_diag2'] = 22
    else:
        df.loc[index, 'level2_diag2'] = 0

    # Repeat the same mapping logic for 'level2_diag3'
    if (row['level2_diag3'] >= 390 and row['level2_diag3'] < 399):
        df.loc[index, 'level2_diag3'] = 1
    elif (row['level2_diag3'] >= 401 and row['level2_diag3'] < 415):
        df.loc[index, 'level2_diag3'] = 2
    elif (row['level2_diag3'] >= 415 and row['level2_diag3'] < 460):
        df.loc[index, 'level2_diag3'] = 3
    elif (np.floor(row['level2_diag3']) == 785):
        df.loc[index, 'level2_diag3'] = 4
    elif (row['level2_diag3'] >= 460 and row['level2_diag3'] < 489):
        df.loc[index, 'level2_diag3'] = 5
    elif (row['level2_diag3'] >= 490 and row['level2_diag3'] < 497):
        df.loc[index, 'level2_diag3'] = 6
    elif (row['level2_diag3'] >= 500 and row['level2_diag3'] < 520):
        df.loc[index, 'level2_diag3'] = 7
    elif (np.floor(row['level2_diag3']) == 786):
        df.loc[index, 'level2_diag3'] = 8
    elif (row['level2_diag3'] >= 520 and row['level2_diag3'] < 530):
        df.loc[index, 'level2_diag3'] = 9
    elif (row['level2_diag3'] >= 530 and row['level2_diag3'] < 544):
        df.loc[index, 'level2_diag3'] = 10
    elif (row['level2_diag3'] >= 550 and row['level2_diag3'] < 554):
        df.loc[index, 'level2_diag3'] = 11
    elif (row['level2_diag3'] >= 555 and row['level2_diag3'] < 580):
        df.loc[index, 'level2_diag3'] = 12
    elif (np.floor(row['level2_diag3']) == 787):
        df.loc[index, 'level2_diag3'] = 13
    elif (np.floor(row['level2_diag3']) == 250):
        df.loc[index, 'level2_diag3'] = 14
    elif (row['level2_diag3'] >= 800 and row['level2_diag3'] < 1000):
        df.loc[index, 'level2_diag3'] = 15
    elif (row['level2_diag3'] >= 710 and row['level2_diag3'] < 740):
        df.loc[index, 'level2_diag3'] = 16
    elif (row['level2_diag3'] >= 580 and row['level2_diag3'] < 630):
        df.loc[index, 'level2_diag3'] = 17
    elif (np.floor(row['level2_diag3']) == 788):
        df.loc[index, 'level2_diag3'] = 18
    elif (row['level2_diag3'] >= 140 and row['level2_diag3'] < 240):
        df.loc[index, 'level2_diag3'] = 19
    elif row['level2_diag3'] >= 240 and row['level2_diag3'] < 280 and (np.floor(row['level2_diag3']) != 250):
        df.loc[index, 'level2_diag3'] = 20
    elif (row['level2_diag3'] >= 680 and row['level2_diag3'] < 710) or (np.floor(row['level2_diag3']) == 782):
        df.loc[index, 'level2_diag3'] = 21
    elif (row['level2_diag3'] >= 290 and row['level2_diag3'] < 320):
        df.loc[index, 'level2_diag3'] = 22
    else:
        df.loc[index, 'level2_diag3'] = 0

print(df[['diag_1','level1_diag1']].head(15).T)

# Save the processed DataFrame as a CSV file
df.to_csv('./diabetes_data_preprocessed.csv')
    