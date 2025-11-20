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



os.environ['PYTENSOR_FLAGS'] = 'optimiser=fast_compile'

# Establish connection to SQL Server
conn = pyodbc.connect(

)

gender = "male"

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

# TODO: remove this after update, it currently randomly add small number to chol_hdl_ratio to prevent "The term is constant!" error until dataset is updated
df['chol_hdl_ratio'] = df['chol_hdl_ratio'] + np.random.uniform(0.01, 0.09, size=df['chol_hdl_ratio'].shape)

# merge chinese into other asian due to the small number of chinese population
df.loc[df['chinese'] == 1, 'other_asian'] = 1
df.drop('chinese', axis=1, inplace=True)
predictors.remove('chinese')

df['bmi'] = df['bmi'].astype(float)
df['townsend_score'] = df['townsend_score'].astype(float)

df['age_bmi'] = df['age']*df['bmi']
df['age_townsend'] = df['age']*df['townsend_score']
df['age_sbp'] = df['age']*df['sbp']
df['age_fh_chd'] = df['age']*df['fh_chd']
df['age_smoking'] = df['age']*df['current_smoker']
df['age_treated_htn'] = df['age']*df['treated_htn']
df['age_diabetes'] = df['age']*df['diabetes']
df['age_af'] = df['age']*df['af']

# convert the date column to datetime
df['date'] = pd.to_datetime(df['date'])


method_str = 'Regular Testing'  # Options: 'Baseline', 'Regular Testing', 'Static Threshold', 'SPC', 'Bayesian'

# Only change these dates when batching for the Bayesian method
#startDate = pd.to_datetime('01-04-2008', dayfirst=True)
startDate = pd.to_datetime('01-04-2008', dayfirst=True)
#endDate = pd.to_datetime('01-07-2008', dayfirst=True)
endDate = pd.to_datetime('19-08-2015', dayfirst=True) # Most recent record minus 10 years: '2025-08-19'

df = df[df['date']<= endDate]

plot_patients_per_month(df, model_type='qrisk2', gender=gender)

# If the startDate and endDate are the full period then fit the initial model
if startDate == pd.to_datetime('01-04-2008', dayfirst=True):

    prior_six_months = df[(df['date'] >= pd.to_datetime(startDate) - relativedelta(months=6) ) & (df['date'] < pd.to_datetime(startDate))]

    X = prior_six_months[predictors + interactions]
    y = prior_six_months[['outcome']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=49)

    model = LogisticRegression(max_iter=2000, solver='saga', random_state=5)
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
    recalthreshold = auroc - (0.1*auroc) # refitted AUROC
    # save the auroc recal threshold for bayesian batching
    if method_str == "Bayesian":
        with open(f"qrisk_{gender}_auroc_thresh.txt", "w") as file:
            file.write(str(recalthreshold))
    print("Original AUROC:", auroc)
    #recalthreshold = 0.811 # Paper has AUROC of 0.814, with lower CI at 0.811 

    if method_str != "Bayesian":
        saved_df = df[['date', 'outcome']]


else: # Else will only trigger when Bayesian model is run as other methods don't require batching dates
    # select most recent coefs from the bayesian df and set them as the coefs
    bayes_coefs_df = pd.read_csv(f'qrisk2_{gender}_Bayesian_coefs.csv')
    with open(f"qrisk_{gender}_auroc_thresh.txt", "r") as file:
        recalthreshold = float(file.read())

    
    bayes_coefs_df['date'] = pd.to_datetime(bayes_coefs_df['date'])
    bayes_coefs_df = bayes_coefs_df.sort_values('date')
    latest_bayes_coef = bayes_coefs_df['date'].max()
    latest_bayes_df = bayes_coefs_df[bayes_coefs_df['date'] == latest_bayes_coef]


    coefs = {}
    coefs_std = {}

    for col in latest_bayes_df.columns:
        if col != 'date':
            # Split "(mean, std)" into two floats
            parsed = latest_bayes_df[col].str.strip("()").str.split(",", expand=True)
            mean_values = parsed[0].astype(float).tolist()[0]
            std_values = parsed[1].astype(float).tolist()[0]

            coefs[col] = mean_values
            coefs_std[col] = std_values

    print(coefs)
    print(coefs_std)

df = df[df['date']>= startDate]

# Calculate baseline log-odds
weighted_coef_sum = coefs['white']*df['white'] + coefs['indian']*df['indian'] + coefs['pakistani']*df['pakistani'] + coefs['bangladeshi']*df['bangladeshi'] 
weighted_coef_sum += coefs['other_asian']*df['other_asian'] + coefs['black_caribbean']*df['black_caribbean'] + coefs['black_african']*df['black_african'] 
weighted_coef_sum += coefs['other_ethnicity']*df['other_ethnicity'] + coefs['age']*(df['age']) + coefs['bmi']*(df['bmi']) 
weighted_coef_sum += coefs['townsend_score']*(df['townsend_score']) + coefs['sbp']*(df['sbp']) + coefs['chol_hdl_ratio']*(df['chol_hdl_ratio']) 
weighted_coef_sum += coefs["fh_chd"]*(df["fh_chd"]) + coefs["current_smoker"]*(df["current_smoker"]) 
weighted_coef_sum += coefs["treated_htn"]*(df["treated_htn"]) + coefs["diabetes"]*(df["diabetes"]) + coefs["ra"]*(df["ra"]) 
weighted_coef_sum += coefs["af"]*(df["af"]) + coefs["ckd"]*(df["ckd"]) + (coefs["age_bmi"] * df["age"] * df["bmi"]) 
weighted_coef_sum += (coefs["age_townsend"] * df["age_townsend"]) + (coefs["age_sbp"] * df["age_sbp"]) 
weighted_coef_sum += (coefs["age_fh_chd"] * df["age_fh_chd"]) + (coefs["age_smoking"] * df["age_smoking"]) 
weighted_coef_sum += (coefs["age_treated_htn"] * df["age_treated_htn"]) + (coefs["age_diabetes"] * df["age_diabetes"])
weighted_coef_sum += (coefs["age_af"] * df["age_af"])


# Compute log-odds
lp = coefs['Intercept'] + weighted_coef_sum
lp = np.clip(lp, -20, 20)  # Clip to avoid overflow issues

# Estimate predictions
curpredictions = 1 / (1 + np.exp(-lp))  # Convert to probability

df['prediction'] = curpredictions


if method_str != 'Bayesian':
    saved_df.to_csv(f'{method_str}_qrisk2_{gender}_predictions_and_outcomes.csv', index=False)


# Baseline Testing
if method_str == 'Baseline':
    model = RecalibratePredictions()
    model.trigger = TimeframeTrigger(model=model, updateTimestep=9_999, dataStart=startDate, dataEnd=endDate)
    mytest = PREDICT(data=df, model=model, startDate=startDate, endDate=endDate, timestep='month')

# Regular Testing
if method_str == 'Regular Testing':
    reg_time = 3*365
    model = RecalibratePredictions()
    model.trigger = TimeframeTrigger(model=model, updateTimestep=reg_time, dataStart=startDate, dataEnd=endDate)
    mytest = PREDICT(data=df, model=model, startDate=startDate, endDate=endDate, timestep='month', recal_period=reg_time)


# Static Threshold Testing
if method_str == 'Static Threshold':
    model = RecalibratePredictions()
    model.trigger = AUROCThreshold(model=model, update_threshold=recalthreshold)
    mytest = PREDICT(data=df, model=model, startDate=startDate, endDate=endDate, timestep='month')


# SPC Testing
if method_str == 'SPC':
    model = RecalibratePredictions()
    model.trigger = SPCTrigger(model=model, input_data=df, numMonths=12, verbose=False)
    mytest = PREDICT(data=df, model=model, startDate=startDate, endDate=endDate, timestep='month')


# Bayesian Testing
if method_str == 'Bayesian':
    model = BayesianModel(input_data=df, priors={"Intercept": (coefs["Intercept"], coefs_std["Intercept"]),
                            "white": (coefs['white'], coefs_std['white']), 
                            "indian": (coefs['indian'], coefs_std['indian']),
                            "pakistani": (coefs['pakistani'], coefs_std['pakistani']),
                            "bangladeshi": (coefs['bangladeshi'], coefs_std['bangladeshi']),
                            "other_asian": (coefs['other_asian'], coefs_std['other_asian']),
                            "black_caribbean": (coefs['black_caribbean'], coefs_std['black_caribbean']),
                            "black_african": (coefs['black_african'], coefs_std['black_african']),
                            "other_ethnicity": (coefs['other_ethnicity'], coefs_std['other_ethnicity']),
                            "age": (coefs['age'], coefs_std['age']),
                            "bmi": (coefs['bmi'], coefs_std['bmi']),
                            "townsend_score": (coefs['townsend_score'], coefs_std['townsend_score']),
                            "sbp": (coefs['sbp'], coefs_std['sbp']),
                            "chol_hdl_ratio": (coefs['chol_hdl_ratio'], coefs_std['chol_hdl_ratio']),
                            "fh_chd": (coefs['fh_chd'], coefs_std['fh_chd']),
                            "current_smoker": (coefs['current_smoker'], coefs_std['current_smoker']),
                            "treated_htn": (coefs['treated_htn'], coefs_std['treated_htn']),
                            "diabetes": (coefs['diabetes'], coefs_std['diabetes']),
                            "ra": (coefs['ra'], coefs_std['ra']),
                            "af": (coefs['af'], coefs_std['af']),
                            "ckd": (coefs['ckd'], coefs_std['ckd']),
                            "age_bmi": (coefs['age_bmi'], coefs_std['age_bmi']),
                            "age_townsend": (coefs['age_townsend'], coefs_std['age_townsend']),
                            "age_sbp": (coefs['age_sbp'], coefs_std['age_sbp']),
                            "age_fh_chd": (coefs['age_fh_chd'], coefs_std['age_fh_chd']),
                            "age_smoking": (coefs['age_smoking'], coefs_std['age_smoking']),
                            "age_treated_htn": (coefs['age_treated_htn'], coefs_std['age_treated_htn']),
                            "age_diabetes": (coefs['age_diabetes'], coefs_std['age_diabetes']),
                            "age_af": (coefs['age_af'], coefs_std['age_af'])}, 
                            cores=1, verbose=False,
                            model_formula="outcome ~ white + indian + pakistani + bangladeshi + other_asian + black_caribbean + black_african + other_ethnicity + age + bmi + townsend_score + sbp + chol_hdl_ratio + fh_chd + current_smoker + treated_htn + diabetes + ra + af + ckd + age_bmi + age_townsend + age_sbp + age_fh_chd + age_smoking + age_treated_htn + age_diabetes + age_af")
    model.trigger = TimeframeTrigger(model=model, updateTimestep='month', dataStart=startDate, dataEnd=endDate)
    mytest = PREDICT(data=df, model=model, startDate=startDate, endDate=endDate, timestep='month', recal_period=30)

############################################ Run Testing ###########################################

mytest.addLogHook(Accuracy(model))
mytest.addLogHook(AUROC(model))
mytest.addLogHook(Precision(model))
mytest.addLogHook(CalibrationSlope(model))
mytest.addLogHook(CITL(model))
mytest.addLogHook(OE(model))
mytest.addLogHook(AUPRC(model))
if method_str == 'Bayesian':
    mytest.addLogHook(TrackBayesianCoefs(model))
mytest.run()
log = mytest.getLog()


if 'Model Updated' in log:
    model_updates = pd.DataFrame({'date': list(log['Model Updated'].keys()), 'method': list([method_str] * len(log['Model Updated'].keys()))})

if method_str == 'Bayesian':
    # if this is the first batch of dates for the bayesian model set up the coefs csv
    if startDate == pd.to_datetime('01-04-2008', dayfirst=True):
        # create and save df with headers
        coef_columns = ['date'] + predictors + interactions + [value + 'std' for value in predictors] + [value + 'std' for value in interactions]

    bayes_coefs_df = pd.DataFrame.from_dict(log["BayesianCoefficients"], orient='index').reset_index()
    bayes_coefs_df.columns = ['date'] + list(bayes_coefs_df.columns[1:])
    
    bayes_coefs_df.to_csv(f'qrisk2_{gender}_Bayesian_coefs.csv', mode='a', index=False)
    bayes_coefs_df = pd.read_csv(f'qrisk2_{gender}_Bayesian_coefs.csv')
    # filter rows where 'date' column doesn't contain 'date'
    rows_to_drop = bayes_coefs_df.apply(lambda row: row.astype(str).str.contains('date').any(), axis=1)
    bayes_coefs_df = bayes_coefs_df.drop(bayes_coefs_df[rows_to_drop].index)
    # delete duplicate Bayesian Coefficients caused by the overlap in dates
    bayes_coefs_df = bayes_coefs_df.drop_duplicates(subset=['date'], keep='last')
    bayes_coefs_df.to_csv('qrisk2_{gender}_Bayesian_coefs.csv', index=False) # save the changes to the csv


    bayes_coefs_df['date'] = pd.to_datetime(bayes_coefs_df['date'])

    BayesianCoefsPlot(bayes_coefs_df, f'qrisk_{gender}')
    bayes_preds_df = pd.DataFrame({'date': list(df['date']), 'outcome': list(df['outcome']), 'Bayesian preds': model.predict(df), 'startDate': list([startDate] * len(df["date"])), 'endDate': list([endDate] * len(df["date"]))})

    bayes_preds_df.to_csv(f'Bayesian_qrisk2_{gender}_predictions_and_outcomes.csv', mode='a', header=False, index=False)

    # remove duplicate prediction rows caused by overlap batching
    if endDate == pd.to_datetime('19-08-2015', dayfirst=True):
        bayes_preds_df = pd.read_csv(f'Bayesian_qrisk2_{gender}_predictions_and_outcomes.csv')
        cols_to_convert = ['date', 'startDate', 'endDate']
        bayes_preds_df[cols_to_convert] = bayes_preds_df[cols_to_convert].apply(pd.to_datetime)

        bayes_preds_df['adjusted_end'] = bayes_preds_df['endDate'].apply(lambda d: d - relativedelta(months=1))

        filtered_preds_df = bayes_preds_df[bayes_preds_df['date'] < bayes_preds_df['adjusted_end']]
        filtered_preds_df.drop(columns=['adjusted_end'], inplace=True)
        filtered_preds_df.to_csv(f'Bayesian_qrisk2_{gender}_predictions_and_outcomes.csv', index=False)



if method_str != 'Bayesian':
    saved_df[method_str + ' preds'] = model.predict(df)

    # if (saved_df[method_str + " preds"] < 0.5).all():
    #     print("all predicted same class")

    saved_df.to_csv(f'{method_str}_qrisk2_{gender}_predictions_and_outcomes.csv', index=False)

########################################### Save Metrics #######################################
metrics = pd.DataFrame({'Time': list(log["Accuracy"].keys()), 'Accuracy': list(log["Accuracy"].values()), 'AUROC': list(log["AUROC"].values()), 'Precision': list(log["Precision"].values()), 'CalibrationSlope': list(log["CalibrationSlope"].values()), 'CITL': list(log["CITL"].values()), 'OE': list(log["O/E"].values()), 'AUPRC': list(log["AUPRC"].values()),'Method':list([method_str] * len(log["Accuracy"]))})

metrics.to_csv(f'qrisk2_{gender}_performance_metrics.csv', mode='a', header=False, index=False)
if 'Model Updated' in log:
    model_updates.to_csv(f'qrisk2_{gender}_model_updates.csv', mode='a', header=False, index=False)



############################################ Plot Metrics #######################################

plot_count_of_patients_over_threshold_risk(threshold=0.1, model_type='qrisk2', gender=gender)

plot_method_comparison_metrics(metrics_df = f'qrisk2_{gender}_performance_metrics.csv', recalthreshold=recalthreshold, 
                               model_updates=f'qrisk2_{gender}_model_updates.csv', model_type='qrisk2', gender=gender)