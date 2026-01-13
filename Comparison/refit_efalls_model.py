import pyodbc
import sys
sys.path.append("../")
from PREDICT import PREDICT
from PREDICT.Models import *
from PREDICT.Metrics import *
from PREDICT.Triggers import *
from PREDICT.Plots import *
import numpy as np
import pandas as pd
from datetime import timedelta
import datetime
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil.relativedelta import relativedelta
import logging
import warnings
warnings.filterwarnings('ignore')
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
import json



os.environ['PYTENSOR_FLAGS'] = 'optimiser=fast_compile'

# Establish connection to SQL Server
conn = pyodbc.connect(
    "DRIVER={SQL Server};"
    "SERVER=BHTS-CONNECTYO3;"
    "DATABASE=CB_2151;"
    "Trusted_Connection=yes;"
)


# Query data from a table
query = f"SELECT * FROM tbl_final_efalls_deduped WHERE DateOnly < '2019-01-01'"

# Store query result in a dataframe
df = pd.read_sql(query, conn)

# Close the connection
conn.close()

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
                "Washing_and_bathing", "Weakness", "Weight_loss"]#, "Intercept"]

# select specific columns
df = df[["Fall_Outcome", "DateOnly"]+predictors]
# change some of the column names
df.rename(columns={"DateOnly": "date", "Fall_Outcome":"outcome"}, inplace=True)
# convert the date column to datetime
df['date'] = pd.to_datetime(df['date'])

# define analysis window
startDate = pd.to_datetime('01-01-2019', dayfirst=True)

# restrict to endDate if present - refit efalls uses records before 2019-01-01 so ensure we filter
# (original selection already queried DateOnly < '2019-01-01')
plot_patients_per_month(df, model_type='efalls')

# select prior six months used to fit scalers
prior_six_months = df[(df['date'] >= startDate - relativedelta(months=6)) & (df['date'] < startDate)]

# fit scaler on prior six months only and apply to entire dataframe
scaler_params = {}
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
arr = prior_six_months[['Age']].astype(float).values
sc.fit(arr)
mean_val = float(sc.mean_[0])
scale_val = float(sc.scale_[0]) if float(sc.scale_[0]) != 0 else 1.0
scaler_params['Age'] = {'mean': mean_val, 'scale': scale_val}
df['Age'] = (df['Age'].astype(float) - mean_val) / scale_val

# persist scaler parameters to JSON
with open('efalls_scaler.json', 'w') as sf:
    json.dump(scaler_params, sf)

# select prior six months for model fitting
prior_six_months = df[(df['date'] >= startDate - relativedelta(months=6)) & (df['date'] < startDate)]

# # TODO: Do we want to scale Polypharmacy too?
# scaled_Polypharmacy = scaler.fit_transform(df[['Polypharmacy']])
# df['Polypharmacy'] = pd.DataFrame(scaled_Polypharmacy, columns=['Polypharmacy'])





startDate = pd.to_datetime('01-01-2019', dayfirst=True)

plot_patients_per_month(df, model_type='efalls')


X = df[predictors]
y = df[['outcome']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

model = LogisticRegression(max_iter=2000, penalty=None)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

intercept = model.intercept_
print("Prefit model Intercept:", intercept)
print("Prefit model coefficients:", model.coef_[0])
print(f"L2 Norm of coefficients: {np.linalg.norm(model.coef_)}")

coefs = dict(zip(predictors, model.coef_.ravel().tolist()))
# add the intercept to the coefs dict
coefs['Intercept'] = float(intercept)
coefs_std = {key: 0.25 for key in coefs} # make all the coef stds 0.25

y_prob = model.predict_proba(X_test)[:, 1]
auroc = roc_auc_score(y_test, y_prob)
# bootstrap AUROC to estimate mean and SD, then set recalthreshold = mean - 1.96*SD
n_boot = 2000
rng = np.random.RandomState(5)
y_true = y_test.values.ravel()
y_scores = y_prob
boot_aucs = []
n = len(y_true)
for _ in range(n_boot):
    idx = rng.randint(0, n, n)
    y_t = y_true[idx]
    y_s = y_scores[idx]
    if y_t.min() == y_t.max():
        continue
    try:
        boot_aucs.append(roc_auc_score(y_t, y_s))
    except ValueError:
        continue

boot_aucs = np.array(boot_aucs)
if boot_aucs.size == 0:
    print('Bootstrap sampling failed.')
    recalthreshold = auroc - (0.1 * auroc)
else:
    mean_auc = float(np.mean(boot_aucs))
    sd_auc = float(np.std(boot_aucs, ddof=1))
    recalthreshold = mean_auc - 1.96 * sd_auc

print(f"Bootstrap AUROC mean: {mean_auc:.6f}, SD: {sd_auc:.6f}, recalthreshold: {recalthreshold:.6f}")
# save the auroc recal threshold for bayesian batching
with open(f"efalls_auroc_thresh.txt", "w") as file:
    file.write(str(recalthreshold))
print("Original AUROC:", auroc)
#recalthreshold = 0.811 # Paper has AUROC of 0.814, with lower CI at 0.811 
# print(len(coefs))
# print(coefs)
with open('efalls_coefs.json', 'w') as f:
    json.dump(coefs, f)