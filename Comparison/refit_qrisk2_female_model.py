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
import statsmodels.api as sm
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

gender = "female"

resultsloc = f'results/qrisk2_{gender}'
os.makedirs(resultsloc, exist_ok=True)

# Query data from a table
query = f"SELECT * FROM qrisk_{gender}s"

# Store query result in a dataframe
df = pd.read_sql(query, conn)

# Close the connection
conn.close()


predictors = ["age", "chol_hdl_ratio", "current_smoker", "bmi", "townsend_score", "sbp", "fh_chd", "treated_htn", "diabetes", "ra", "af", "ckd", "bangladeshi", "chinese", "indian", "other_asian", "pakistani", "black_african", "black_caribbean", "other_ethnicity", "white"]
interactions = ["age_bmi", "age_townsend", "age_sbp", "age_fh_chd", "age_smoking", "age_treated_htn", "age_diabetes", "age_af"]

# select specific columns
df = df[["outcome", "DateOnly", "Age", "chol_hdl_ratio", "smoker_status", "bmi", "townsend_score", "sbp", "fh_chd", "treated_htn", "diabetes", "ra", "af", "ckd", "bangladeshi", "chinese", "indian", "other_asian", "pakistani", "black_african", "black_caribbean", "other_ethnicity", "white"]]
# change some of the column names
df.rename(columns={"DateOnly": "date", "smoker_status": "current_smoker", "Age": "age"}, inplace=True)
print(df.head())
# TODO: remove this after update, it currently randomly add small number to chol_hdl_ratio to prevent "The term is constant!" error until dataset is updated
df['chol_hdl_ratio'] = df['chol_hdl_ratio'] + np.random.normal(0.0, 0.01, size=df['chol_hdl_ratio'].shape)

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
endDate = pd.to_datetime('19-08-2015', dayfirst=True) # Most recent record minus 10 years: '2025-08-19'

# restrict to endDate and plot
df = df[df['date']<= endDate]
plot_patients_per_month(df, model_type='qrisk2', gender=gender, resultsloc=resultsloc)

# select the prior six months used to fit the prefit model and the scalers
prior_six_months = df[(df['date'] >= startDate - relativedelta(months=6)) & (df['date'] < startDate)]

# fit scalers on the prior six months only, apply to entire dataframe, and save parameters
scaler_params = {}
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
for var in ['age', 'chol_hdl_ratio', 'bmi', 'townsend_score']:
    sc = StandardScaler()
    arr = prior_six_months[[var]].astype(float).values
    sc.fit(arr)
    mean_val = float(sc.mean_[0])
    scale_val = float(sc.scale_[0]) if float(sc.scale_[0]) != 0 else 1.0
    scaler_params[var] = {'mean': mean_val, 'scale': scale_val}
    # apply scaling to full dataframe
    df[var] = (df[var].astype(float) - mean_val) / scale_val

# persist scaler parameters to JSON for downstream scripts
with open(os.path.join(resultsloc, f'qrisk2_{gender}_scaler.json'), 'w') as sf:
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

# select the prior six months used to fit the prefit model and the scalers
prior_six_months = df[(df['date'] >= pd.to_datetime(startDate) - relativedelta(months=6) ) & (df['date'] < pd.to_datetime(startDate))]

X = prior_six_months[predictors + interactions]
y = prior_six_months['outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=5)

model = LogisticRegression(penalty=None, solver='lbfgs', tol=1e-8, max_iter=20000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

intercept = model.intercept_
print("Prefit model Intercept:", intercept)
print("Prefit model coefficients:", model.coef_[0])
print(f"L2 Norm of coefficients: {np.linalg.norm(model.coef_)}")

X_train_sm = sm.add_constant(X_train)
sm_model = sm.GLM(y_train, X_train_sm, family=sm.families.Binomial())
sm_results = sm_model.fit()

coefs = dict(zip(predictors + interactions, sm_results.params[1:].tolist()))
coefs['Intercept'] = float(sm_results.params[0])

coefs_std = dict(zip(predictors + interactions, sm_results.bse[1:].tolist()))
coefs_std['Intercept'] = float(sm_results.bse[0])


X_test_sm = sm.add_constant(X_test)
y_prob = sm_results.predict(X_test_sm).values
auroc = roc_auc_score(y_test, y_prob)
# bootstrap AUROC, calibration-in-the-large (CITL) and calibration slope; estimate mean and SD
n_boot = 500
rng = np.random.RandomState(5)
y_true = y_test.values.ravel()
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

with open(os.path.join(resultsloc, f"qrisk_{gender}_thresh.json"), "w") as jf:
    json.dump(out, jf, indent=2)

with open(os.path.join(resultsloc, f"qrisk2_{gender}_coefs.json"), "w") as f:
    json.dump(coefs, f)
    
with open(os.path.join(resultsloc, f"qrisk2_{gender}_coefs_std.json"), "w") as f:
    json.dump(coefs_std, f)
