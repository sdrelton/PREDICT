import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pyodbc
import sys
sys.path.append("../")
import pandas as pd
from datetime import timedelta
import datetime
from dateutil.relativedelta import relativedelta
import logging
import warnings
warnings.filterwarnings('ignore')
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import json
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



# Establish connection to SQL Server
conn = pyodbc.connect(
    "DRIVER={SQL Server};"
    "Trusted_Connection=yes;"
)


model_name = "tbl_final_efalls_deduped" #"qrisk_males", "qrisk_females", "tbl_final_efalls_deduped"
term = "short_term" # "short_term", "long_term"
# Query data from a table
query = f"SELECT * FROM {model_name}"



if model_name == "tbl_final_efalls_deduped":
    model_name = "efalls"
    startDate = pd.to_datetime('01-01-2019', dayfirst=True)
    endDate = pd.to_datetime('19-08-2024', dayfirst=True) 
else:
    startDate = pd.to_datetime('01-04-2008', dayfirst=True)
    endDate = pd.to_datetime('19-08-2015', dayfirst=True)

# Store query result in a dataframe
df = pd.read_sql(query, conn)

# Close the connection
conn.close()

if model_name != "efalls":
    predictors = ["age", "chol_hdl_ratio", "current_smoker", "bmi", "townsend_score", "sbp", "fh_chd", "treated_htn", "diabetes", "ra", "af", "ckd", "bangladeshi", "chinese", "indian", "other_asian", "pakistani", "black_african", "black_caribbean", "other_ethnicity", "white"]

    # select specific columns
    df = df[["DateOnly", "Age", "chol_hdl_ratio", "smoker_status", "bmi", "townsend_score", "sbp", "fh_chd", "treated_htn", "diabetes", "ra", "af", "ckd", "bangladeshi", "chinese", "indian", "other_asian", "pakistani", "black_african", "black_caribbean", "other_ethnicity", "white"]]
    # change some of the column names
    df.rename(columns={"DateOnly": "date", "smoker_status": "current_smoker", "Age": "age"}, inplace=True)

    # merge chinese into other asian due to the small number of chinese population
    df.loc[df['chinese'] == 1, 'other_asian'] = 1
    df.drop('chinese', axis=1, inplace=True)
    predictors.remove('chinese')

    df['bmi'] = df['bmi'].astype(float)
    df['townsend_score'] = df['townsend_score'].astype(float)
    df['townsend_score_unscaled'] = df['townsend_score'].copy()

    # scale continuous variables:
    scaler = StandardScaler()

    scaled_age = scaler.fit_transform(df[['age']])
    df['age'] = pd.DataFrame(scaled_age, columns=['age'])
    scaled_chol_hdl_ratio = scaler.fit_transform(df[['chol_hdl_ratio']])
    df['chol_hdl_ratio'] = pd.DataFrame(scaled_chol_hdl_ratio, columns=['chol_hdl_ratio'])
    scaled_bmi = scaler.fit_transform(df[['bmi']])
    df['bmi'] = pd.DataFrame(scaled_bmi, columns=['bmi'])
    scaled_townsend = scaler.fit_transform(df[['townsend_score']])
    df['townsend_score'] = pd.DataFrame(scaled_townsend, columns=['townsend_score'])
else:
    df['unique_bnf_last_3_months'] = df['unique_bnf_last_3_months'].astype(float)
    df["Polypharmacy"] = np.log(df["unique_bnf_last_3_months"] + 1) / 10 

    predictors = ["Age", "Female", "Polypharmacy", "Underweight", "Normal", "Obese", "BMI_missing", "Current_Smoker", "Harmful_drinking",
                    "Higher_risk_drinking", "Previous_harmful_drinking", "Zero_Alcohol", "Alcohol_missing", "Abdominal_pain", "Activity_limitation", 
                    "Anaemia_and_haematinic_deficiency", "Asthma", "Atrial_fibrillation", "Back_pain", "Bone_disease", "Cancer", "Cognitive_impairment",
                    "COPD", "Dementia", "Depression", "Diabetes_mellitus", "Dizziness", "Dressing_and_grooming_problems", "Faecal_incontinence",
                    "Falls", "Fatigue", "Foot_problems", "Fracture", "Fragility_fracture", "General_mental_health", "Headache",
                    "Hearing_impairment", "Heart_failure", "Housebound", "Hypertension", "Hypotension_or_syncope", "Inflammatory_arthritis", 
                    "Inflammatory_bowel_disease", "Liver_problems", "Meal_preparation_problems", "Medication_management",
                    "Memory_concerns", "Mobility_problems", "Mono_or_hemiparesis", "Motor_neurone_disease", "Musculoskeletal_problems", "Osteoarthritis", 
                    "Osteoporosis", "Palliative_care", "Parkinsonism_and_tremor", "Peptic_ulcer_disease", "Peripheral_neuropathy", "Peripheral_vascular_disease",
                    "Requirement_for_care", "Respiratory_disease", "Seizures", "Self_harm", "Severe_mental_illness", "Skin_ulcer", "Sleep_problems",
                    "Social_vulnerability", "Stress", "Stroke", "Thyroid_problems", "Urinary_incontinence", "Urinary_system_disease", "Visual_impairment",
                    "Washing_and_bathing", "Weakness", "Weight_loss"]


    df = df[["DateOnly"]+predictors]

    df.rename(columns={"DateOnly": "date", "Fall_Outcome":"outcome"}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df["Female"] = df["Female"].astype(int)
    df["Current_Smoker"] = df["Current_Smoker"].astype(int)

    # scale continuous variables:
    scaler = StandardScaler()

    scaled_age = scaler.fit_transform(df[['Age']])
    df['Age'] = pd.DataFrame(scaled_age, columns=['Age'])

df['date'] = pd.to_datetime(df['date'])


num_months = relativedelta(endDate, startDate).years * 12 + relativedelta(endDate, startDate).months

# This work uses the ideas from https://link.springer.com/chapter/10.1007/978-3-031-58547-0_14

# compare month to month - short term drift
# compare first month to each month after - long term drift

date = []
prediction_drift = []
for month_num in range(0, num_months-3):
    if term == "short_term":
        date1 = pd.to_datetime(startDate) + relativedelta(months=month_num)
        date2 = pd.to_datetime(startDate)  + relativedelta(months=month_num+1)
    if term == "long_term":
        date1 = pd.to_datetime(startDate) + relativedelta(months=0)
        date2 = pd.to_datetime(startDate)  + relativedelta(months=1)
    date3 = pd.to_datetime(startDate) + relativedelta(months=month_num+2)
    date4 = pd.to_datetime(startDate)  + relativedelta(months=month_num+3)

    first_month = df[(df['date'] >= date1 ) & (df['date'] < date2)]
    second_month = df[(df['date'] >= date3 ) & (df['date'] < date4)]

    Xs = first_month[predictors]
    Xs['dataset'] = 0
    ys = Xs['dataset']

    Xt = second_month[predictors]
    Xt['dataset'] = 1
    yt = Xt['dataset']

    X = pd.concat([Xt, Xs])
    y = pd.concat([ys, yt])


    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    n_samples = len(X_holdout)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    # split the holdout set back into the source and target groups
    X_holdout_s = X_holdout[X_holdout['dataset'] == 0]
    X_holdout_t = X_holdout[X_holdout['dataset'] == 1]


    p_source = clf.predict_proba(X_holdout_s)[:, 1]  # p(y=1 | x in source)
    p_target = clf.predict_proba(X_holdout_t)[:, 0]  # p(y=0 | x in target)

    DE = (1/n_samples) * np.sum(p_source) + (1/n_samples) * np.sum(p_target)

    date.append(date3.strftime('%B %Y'))
    prediction_drift.append(DE)

    #print(f"Domain Discrepancy Estimate (DE): {DE:.4f}. \nNote: A larger number (closer to 1) means more error in the model as it can't discriminate, suggesting less data drift.")


    # ########################### PLOT PCA DISTRIBUTIONS ###########################
    # pca = PCA(n_components=2)
    # X_reduced = pca.fit_transform(X.drop(columns=['dataset']))
    # X_plot = pd.DataFrame(X_reduced, columns=['PC1', 'PC2'])
    # X_plot['dataset'] = y.values

    # plt.figure(figsize=(8, 6))

    # plt.scatter(X_plot[X_plot['dataset'] == 0]['PC1'],
    #             X_plot[X_plot['dataset'] == 0]['PC2'],
    #             alpha=0.5, label='Source (dataset=0)', color='blue')

    # plt.scatter(X_plot[X_plot['dataset'] == 1]['PC1'],
    #             X_plot[X_plot['dataset'] == 1]['PC2'],
    #             alpha=0.5, label='Target (dataset=1)', color='red')

    # plt.title('Source vs Target Domain Distributions (PCA projection) Overlapping Shows Less Data Drift')
    # plt.xlabel('Principal Component 1')
    # plt.ylabel('Principal Component 2')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig("../docs/images/domain_distributions.png")
    # plt.show()
plt.figure(figsize=(12, 6))
plt.plot(date, prediction_drift, marker=None, linestyle='-')
plt.xlabel('Target Month')
plt.xticks(rotation=90)
plt.ylabel('Shift Prediction (Discrimination Error-based)')
plt.title(f"Comparing Input Data Month to Month ({term.replace('_',' ')} drift)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f'../docs/images/predictor_distributions/{model_name}_{term}_input_drift_detection.png')   # saves the figure to disk
plt.show()