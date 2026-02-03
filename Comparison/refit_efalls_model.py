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
from experiment_plots import *

os.environ['PYTENSOR_FLAGS'] = 'optimiser=fast_compile'

# Establish connection to SQL Server
conn = pyodbc.connect(
    "DRIVER={SQL Server};"
    "SERVER=BHTS-CONNECTYO3;"
    "DATABASE=CB_2151;"
    "Trusted_Connection=yes;"
)

resultsloc = f'results/efalls'
os.makedirs(resultsloc, exist_ok=True)

# Query data from a table
query = f"SELECT * FROM tbl_final_efalls_deduped WHERE DateOnly >= '2019-01-01'"

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
plot_patients_per_month(df, model_type='efalls', resultsloc=resultsloc)

# select prior twelve months used to fit scalers
prior_twelve_months = df[(df['date'] >= startDate) & (df['date'] < startDate + relativedelta(months=12))]

# fit scaler on prior twelve months only and apply to entire dataframe
""" scaler_params = {}
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
arr = prior_twelve_months[['Age']].astype(float).values
sc.fit(arr)
mean_val = float(sc.mean_[0])
scale_val = float(sc.scale_[0]) if float(sc.scale_[0]) != 0 else 1.0
scaler_params['Age'] = {'mean': mean_val, 'scale': scale_val}
df['Age'] = (df['Age'].astype(float) - mean_val) / scale_val """

""" # persist scaler parameters to JSON
with open(os.path.join(resultsloc, f'efalls_scaler.json'), 'w') as sf:
    json.dump(scaler_params, sf) """


startDate = pd.to_datetime('01-01-2019', dayfirst=True)

plot_patients_per_month(df, model_type='efalls',  resultsloc=resultsloc)


X = prior_twelve_months[predictors].astype(float)
y = prior_twelve_months['outcome'].astype(float)

""" X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=5)

model = LogisticRegression(penalty=None, solver='lbfgs', tol=1e-8, max_iter=20000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

intercept = model.intercept_
print("Prefit model Intercept:", intercept)
print("Prefit model coefficients:", model.coef_[0])
print(f"L2 Norm of coefficients: {np.linalg.norm(model.coef_)}")

X_train_sm = sm.add_constant(X_train.astype(float))
sm_model = sm.GLM(y_train, X_train_sm, family=sm.families.Binomial())
sm_results = sm_model.fit()

coefs = dict(zip(predictors, sm_results.params[1:].tolist()))
coefs['Intercept'] = float(sm_results.params[0])

coefs_std = dict(zip(predictors, sm_results.bse[1:].tolist()))
coefs_std['Intercept'] = float(sm_results.bse[0])

X_test_sm = sm.add_constant(X_test.astype(float))
y_prob = sm_results.predict(X_test_sm).values """

# Recalibrate efalls model on prior 12 months of data
with open(os.path.join(resultsloc, "efalls_coefs_official.json"), 'r') as f:
    coefs_official = json.load(f)

coef_vector = np.array([coefs_official[f] for f in predictors], dtype=float)    
# Dot product gives weighted sum for each row in df
weighted_coef_sum = np.dot(X, coef_vector)
official_intercept = coefs_official['Intercept']
lp = official_intercept + weighted_coef_sum
recal = sm.GLM(y, sm.add_constant(X), family=sm.families.Binomial()).fit()
y_prob = recal.predict()

print('recal intercept = ', recal.params[0])
print('recal slope = ', recal.params[1])

coefs = {x: coefs_official[x]*recal.params[1] for x in predictors}
coefs['Intercept'] = recal.params[0] + official_intercept*recal.params[1]

auroc = roc_auc_score(y, y_prob)
# bootstrap AUROC, calibration-in-the-large (CITL) and calibration slope; estimate mean and SD
n_boot = 500
rng = np.random.RandomState(5)
y_true = y.values.ravel()
y_scores = y_prob
boot_aucs = []
boot_citls = []
boot_slopes = []
n = len(y_true)
eps = 1e-6  # for clipping probabilities when computing logit


for _ in range(n_boot):
    idx = rng.randint(0, n, n)
    y_t = y_true[idx]
    y_s = y_scores[idx]
    # need both classes present to compute metrics and fit calibration model
    if y_t.min() == y_t.max():
        continue
    try:
        auc = roc_auc_score(y_t, y_s)
    except ValueError:
        continue
    # compute calibration slope and intercept by fitting logistic regression of y ~ logit(p)
    p_clipped = np.clip(y_s, eps, 1 - eps)
    logit_p = np.log(p_clipped / (1 - p_clipped)).reshape(-1, 1)
    try:
        # fit slope using sklearn logistic regression on logit(p)
        calib = LogisticRegression(penalty=None, solver='lbfgs', max_iter=5000)
        calib.fit(logit_p, y_t)
        slope = float(calib.coef_.ravel()[0])
        # compute CITL using statsmodels GLM with offset=logit_p (intercept-only model)
        lp = logit_p.ravel()
        X_sm = np.ones((len(lp), 1))
        model = sm.GLM(y_t, X_sm, family=sm.families.Binomial(), offset=lp)
        result = model.fit()
        citl = float(result.params[0])
    except Exception:
        # skip this bootstrap sample if calibration fit fails
        continue

    boot_aucs.append(auc)
    boot_citls.append(citl)
    boot_slopes.append(slope)

boot_aucs = np.array(boot_aucs)
boot_citls = np.array(boot_citls)
boot_slopes = np.array(boot_slopes)

mean_auc = float(np.mean(boot_aucs))
sd_auc = float(np.std(boot_aucs, ddof=1))
recalthreshold_auc = float(np.percentile(boot_aucs, 2.5))

mean_citl = float(np.mean(boot_citls))
sd_citl = float(np.std(boot_citls, ddof=1))
recalthreshold_citl_lower = float(np.percentile(boot_citls, 2.5))
recalthreshold_citl_upper = float(np.percentile(boot_citls, 97.5))

mean_slope = float(np.mean(boot_slopes))
sd_slope = float(np.std(boot_slopes, ddof=1))
recalthreshold_slope_lower = float(np.percentile(boot_slopes, 2.5))
recalthreshold_slope_upper = float(np.percentile(boot_slopes, 97.5))

print(f"Bootstrap AUROC mean: {mean_auc:.6f}, SD: {sd_auc:.6f}, recalthreshold_auc: {recalthreshold_auc:.6f}")
print(f"Bootstrap CITL mean: {mean_citl:.6f}, SD: {sd_citl:.6f}, recalthreshold_citl: {recalthreshold_citl_lower:.6f} - {recalthreshold_citl_upper:.6f}")
print(f"Bootstrap calibration slope mean: {mean_slope:.6f}, SD: {sd_slope:.6f}, recalthreshold_slope: {recalthreshold_slope_lower:.6f} - {recalthreshold_slope_upper:.6f}")

# save thresholds and calibration summaries to JSON and text files
out = {
    "auroc_mean": mean_auc,
    "auroc_sd": sd_auc,
    "auroc_recalthreshold": recalthreshold_auc,
    "citl_mean": mean_citl,
    "citl_sd": sd_citl,
    "citl_recalthreshold_lower": recalthreshold_citl_lower,
    "citl_recalthreshold_upper": recalthreshold_citl_upper,
    "slope_mean": mean_slope,
    "slope_sd": sd_slope,
    "slope_recalthreshold_lower": recalthreshold_slope_lower,
    "slope_recalthreshold_upper": recalthreshold_slope_upper,
    "bootstrap_samples_used": int(boot_aucs.size)
}

with open(os.path.join(resultsloc, f"efalls_thresh.json"), "w") as jf:
    json.dump(out, jf, indent=2)

with open(os.path.join(resultsloc, f"efalls_coefs.json"), "w") as f:
    json.dump(coefs, f)

#with open(os.path.join(resultsloc, f"efalls_coefs_std.json"), "w") as f:
#    json.dump(coefs_std, f)