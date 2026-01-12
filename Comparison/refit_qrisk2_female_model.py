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

gender = "female"

# Query data from a table
query = f"SELECT * FROM qrisk_{gender}s"

# Store query result in a dataframe
df = pd.read_sql(query, conn)

# Close the connection
conn.close()

# print(len(df))

# # patients can only have one record per month

# df['DateOnly'] = pd.to_datetime(df['DateOnly'])

# # create year-month key
# df['year_month'] = df['DateOnly'].dt.to_period('M')

# # sort newest first and keep first row per person_id + month
# df_sorted = df.sort_values(['person_id','year_month','DateOnly'], ascending=[True, True, False])
# df = df_sorted.drop_duplicates(subset=['person_id','year_month'], keep='first').reset_index(drop=True)
# print(len(df))


predictors = ["age", "chol_hdl_ratio", "current_smoker", "bmi", "townsend_score", "sbp", "fh_chd", "treated_htn", "diabetes", "ra", "af", "ckd", "bangladeshi", "chinese", "indian", "other_asian", "pakistani", "black_african", "black_caribbean", "other_ethnicity", "white"]
interactions = ["age_bmi", "age_townsend", "age_sbp", "age_fh_chd", "age_smoking", "age_treated_htn", "age_diabetes", "age_af"]

# select specific columns
df = df[["outcome", "DateOnly", "Age", "chol_hdl_ratio", "smoker_status", "bmi", "townsend_score", "sbp", "fh_chd", "treated_htn", "diabetes", "ra", "af", "ckd", "bangladeshi", "chinese", "indian", "other_asian", "pakistani", "black_african", "black_caribbean", "other_ethnicity", "white"]]
# change some of the column names
df.rename(columns={"DateOnly": "date", "smoker_status": "current_smoker", "Age": "age"}, inplace=True)


# merge chinese into other asian due to the small number of chinese population
df.loc[df['chinese'] == 1, 'other_asian'] = 1
df.drop('chinese', axis=1, inplace=True)
predictors.remove('chinese')

df['bmi'] = df['bmi'].astype(float)
df['townsend_score'] = df['townsend_score'].astype(float)

# convert the date column to datetime
df['date'] = pd.to_datetime(df['date'])

# define analysis window
startDate = pd.to_datetime('01-04-2008', dayfirst=True)
endDate = pd.to_datetime('19-08-2015', dayfirst=True)

# restrict to endDate and plot
df = df[df['date'] <= endDate]
plot_patients_per_month(df, model_type='qrisk2', gender=gender)

# select the prior six months used to fit the prefit model and the scalers
prior_six_months = df[(df['date'] >= startDate - relativedelta(months=6)) & (df['date'] < startDate)]

# fit scalers on the prior six months only, apply to entire dataframe, and save parameters
scaler_params = {}
from sklearn.preprocessing import StandardScaler
for var in ['age', 'chol_hdl_ratio', 'bmi', 'townsend_score']:
    sc = StandardScaler()
    arr = prior_six_months[[var]].astype(float).values
    sc.fit(arr)
    mean_val = float(sc.mean_[0])
    scale_val = float(sc.scale_[0]) if float(sc.scale_[0]) != 0 else 1.0
    scaler_params[var] = {'mean': mean_val, 'scale': scale_val}
    df[var] = (df[var].astype(float) - mean_val) / scale_val

# persist scaler parameters to JSON for downstream scripts
with open(f'qrisk2_{gender}_scaler.json', 'w') as sf:
    json.dump(scaler_params, sf)

# create interaction terms after scaling
df['age_bmi'] = df['age']*df['bmi']
df['age_townsend'] = df['age']*df['townsend_score']
df['age_sbp'] = df['age']*df['sbp']
df['age_fh_chd'] = df['age']*df['fh_chd']
df['age_smoking'] = df['age']*df['current_smoker']
df['age_treated_htn'] = df['age']*df['treated_htn']
df['age_diabetes'] = df['age']*df['diabetes']
df['age_af'] = df['age']*df['af']

# select the prior six months for model fitting
prior_six_months = df[(df['date'] >= startDate - relativedelta(months=6)) & (df['date'] < startDate)]

X = prior_six_months[predictors + interactions]
y = prior_six_months[['outcome']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=5)

model = LogisticRegression(max_iter=2000, penalty=None)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

intercept = model.intercept_
print("Prefit model Intercept:", intercept)
print("Prefit model coefficients:", model.coef_[0])
print(f"L2 Norm of coefficients: {np.linalg.norm(model.coef_)}")

coefs = dict(zip(predictors + interactions, model.coef_.ravel().tolist()))
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
with open(f"qrisk_{gender}_auroc_thresh.txt", "w") as file:
    file.write(str(recalthreshold))
print("Original AUROC:", auroc)
#recalthreshold = 0.811 # Paper has AUROC of 0.814, with lower CI at 0.811 
# print(len(coefs))
# print(coefs)
with open('qrisk2_female_coefs.json', 'w') as f:
    json.dump(coefs, f)