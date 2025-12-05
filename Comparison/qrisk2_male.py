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
from dateutil.relativedelta import relativedelta
import logging
import warnings
warnings.filterwarnings('ignore')
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import json
from sklearn.preprocessing import StandardScaler



os.environ['PYTENSOR_FLAGS'] = 'optimiser=fast_compile'

# Establish connection to SQL Server
conn = pyodbc.connect(
    "DRIVER={SQL Server};"
    "SERVER=BHTS-CONNECTYO3;"
    "DATABASE=CB_2151;"
    "Trusted_Connection=yes;"
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

# merge chinese into other asian due to the small number of chinese population
df.loc[df['chinese'] == 1, 'other_asian'] = 1
df.drop('chinese', axis=1, inplace=True)
predictors.remove('chinese')

df['bmi'] = df['bmi'].astype(float)
df['townsend_score'] = df['townsend_score'].astype(float)

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

# if a file to store the model update dates or performance metrics doesn't exist, then create them with the correct headers
if not os.path.exists(f"qrisk2_{gender}_performance_metrics.csv"):
    perform_metrics_df = pd.DataFrame(columns=['Time','Accuracy','AUROC','Precision','CalibrationSlope','CITL','OE','AUPRC','Method'])
    perform_metrics_df.to_csv(f"qrisk2_{gender}_performance_metrics.csv", index=False)

if not os.path.exists(f"qrisk2_{gender}_model_updates.csv"):
    perform_metrics_df = pd.DataFrame(columns=['date', 'method'])
    perform_metrics_df.to_csv(f"qrisk2_{gender}_model_updates.csv", index=False)

################################## FOR SIMPLICITY RUN THE BAYESIAN METHOD SEPARATELY TO THE OTHER METHODS ##################################
method_strs = ['Baseline', 'Regular Testing', 'Static Threshold', 'SPC']
#method_strs = ['Bayesian']

for method_str in method_strs:

    print(f"Running QRISK {gender} Model using {method_str} method...")
    startOfAnalysis = pd.to_datetime('01-04-2008', dayfirst=True)

    # Only change these dates when batching for the Bayesian method
    startDate = pd.to_datetime('01-04-2008', dayfirst=True)
    #startDate = pd.to_datetime('01-06-2008', dayfirst=True)
    #endDate = pd.to_datetime('01-08-2008', dayfirst=True)
    endDate = pd.to_datetime('19-08-2015', dayfirst=True) # Most recent record minus 10 years: '2025-08-19'

    df = df[df['date']<= endDate]

    plot_patients_per_month(df, model_type='qrisk2', gender=gender)
    with open(f"qrisk_{gender}_auroc_thresh.txt", "r") as file:
        recalthreshold = float(file.read())

    # If the startDate and endDate are the full period then fit the initial model
    if startDate == pd.to_datetime('01-04-2008', dayfirst=True):
        with open(f'qrisk2_{gender}_coefs.json', 'r') as f:
            coefs = json.load(f)

        coefs_std = {key: 0.25 for key in coefs}

    else: # Else will only trigger when Bayesian model is run as other methods don't require batching dates
        # select most recent coefs from the bayesian df and set them as the coefs
        bayes_coefs_df = pd.read_csv(f'qrisk2_{gender}_Bayesian_coefs.csv')
        
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

    df = df[df['date']>= startDate]

    # Calculate baseline log-odds
    coef_vector = np.array([coefs[f] for f in predictors + interactions])
    feature_matrix = np.column_stack([df[f] for f in predictors + interactions])
    # Dot product gives weighted sum for each row in df
    weighted_coef_sum = np.dot(feature_matrix, coef_vector)


    # Compute log-odds
    lp = coefs['Intercept'] + weighted_coef_sum
    lp = np.clip(lp, -20, 20)  # Clip to avoid overflow issues

    # Estimate predictions
    curpredictions = 1 / (1 + np.exp(-lp))  # Convert to probability
    df['prediction'] = curpredictions

    # clear the performance metrics and model updates for the method so we don't duplicate the date
    if (method_str != 'Bayesian') or (method_str=='Bayesian' and startDate == startOfAnalysis):
        org_metrics = pd.read_csv(f'qrisk2_{gender}_performance_metrics.csv')
        org_metrics = org_metrics[org_metrics["Method"] != method_str]
        org_metrics.to_csv(f'qrisk2_{gender}_performance_metrics.csv', mode='w', header=True, index=False)

        org_updates = pd.read_csv(f'qrisk2_{gender}_model_updates.csv')
        org_updates = org_updates[org_updates["method"] != method_str]
        org_updates.to_csv(f'qrisk2_{gender}_model_updates.csv', mode='w', header=True, index=False)

    # Baseline Testing
    if method_str == 'Baseline':
        model = RecalibratePredictions()
        model.trigger = TimeframeTrigger(model=model, updateTimestep=9_999, dataStart=startDate, dataEnd=endDate)
        mytest = PREDICT(data=df, model=model, startDate=startDate, endDate=endDate, timestep='month', model_name=f'{method_str}_qrisk2_{gender}')

    # Regular Testing
    if method_str == 'Regular Testing':
        reg_time = 3*365
        model = RecalibratePredictions()
        model.trigger = TimeframeTrigger(model=model, updateTimestep=reg_time, dataStart=startDate, dataEnd=endDate)
        mytest = PREDICT(data=df, model=model, startDate=startDate, endDate=endDate, timestep='month', recal_period=reg_time, model_name=f'{method_str}_qrisk2_{gender}')


    # Static Threshold Testing
    if method_str == 'Static Threshold':
        model = RecalibratePredictions()
        model.trigger = AUROCThreshold(model=model, update_threshold=recalthreshold)
        mytest = PREDICT(data=df, model=model, startDate=startDate, endDate=endDate, timestep='month', recal_period=365, model_name=f'{method_str}_qrisk2_{gender}')


    # SPC Testing
    if method_str == 'SPC':
        model = RecalibratePredictions()
        model.trigger = SPCTrigger(model=model, input_data=df, numMonths=12, verbose=False)
        mytest = PREDICT(data=df, model=model, startDate=startDate, endDate=endDate, timestep='month', recal_period=365, model_name=f'{method_str}_qrisk2_{gender}')


    # Bayesian Testing
    if method_str == 'Bayesian':
        priors_dict = {"Intercept": (coefs["Intercept"], coefs_std["Intercept"])}
        for predictor in predictors + interactions:
            priors_dict[predictor] = (coefs[predictor], coefs_std[predictor])
        
        # add the initial coefficients to the csv file to plot Bayesian coefficients over time
        if startDate == pd.to_datetime('01-04-2008', dayfirst=True):
            priors_df = pd.DataFrame([priors_dict])
            # add the date
            priors_df.insert(0, 'date', startDate)
            priors_df.to_csv(f'qrisk2_{gender}_Bayesian_coefs.csv', mode='w', index=False, header=True)
            
        model = BayesianModel(input_data=df, priors=priors_dict, 
                                cores=1, draws=100, tune=100, chains=2, verbose=False,
                                model_formula="outcome ~ white + indian + pakistani + bangladeshi + other_asian + black_caribbean + black_african + other_ethnicity + age + bmi + townsend_score + sbp + chol_hdl_ratio + fh_chd + current_smoker + treated_htn + diabetes + ra + af + ckd + age_bmi + age_townsend + age_sbp + age_fh_chd + age_smoking + age_treated_htn + age_diabetes + age_af")
        model.trigger = TimeframeTrigger(model=model, updateTimestep='month', dataStart=startDate, dataEnd=endDate)
        mytest = PREDICT(data=df, model=model, startDate=startDate, endDate=endDate, timestep='month', recal_period=30, model_name=f'{method_str}_qrisk2_{gender}', startOfAnalysis=startOfAnalysis)

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

        if 'Bayesian' not in method_strs:

            # ensure dates are datetimes and updates are sorted
            postPredDf = df[['date', 'outcome', 'prediction']].copy()
            postPredDf['date'] = pd.to_datetime(postPredDf['date'])
            model_update_dates = sorted(pd.to_datetime(list(log['Model Updated'].keys())))

            predictionHooks = model.postPredictHooks

            for i, update_date in enumerate(model_update_dates):
                mask = postPredDf['date'] >= update_date

                # ensure hook returns a Series or array of same length
                original = postPredDf.loc[mask, 'prediction']
                transformed = predictionHooks[i](original)
                postPredDf.loc[mask, 'prediction'] = transformed

            postPredDf.to_csv(f"probs_and_outcomes/{method_str}_qrisk2_{gender}_predictions_and_outcomes.csv", mode='w', header=True, index=False)

    else: # if the model doesn't update
        noUpdatePreds = df[['date', 'outcome', 'prediction']].copy()
        noUpdatePreds.to_csv(f"probs_and_outcomes/{method_str}_qrisk2_{gender}_predictions_and_outcomes.csv", mode='w', header=True, index=False)


    if method_str == 'Bayesian':
        # if this is the first batch of dates for the bayesian model set up the coefs csv
        if startDate == pd.to_datetime('01-04-2008', dayfirst=True):
            # create and save df with headers
            coef_columns = ['date'] + predictors + interactions + [value + 'std' for value in predictors] + [value + 'std' for value in interactions]

        bayes_coefs_df = pd.DataFrame.from_dict(log["BayesianCoefficients"], orient='index').reset_index()
        bayes_coefs_df.columns = ['date'] + list(bayes_coefs_df.columns[1:])
        
        bayes_coefs_df.to_csv(f'qrisk2_{gender}_Bayesian_coefs.csv', mode='a', index=False, header=False)
        bayes_coefs_df = pd.read_csv(f'qrisk2_{gender}_Bayesian_coefs.csv')
        bayes_coefs_df = bayes_coefs_df.drop_duplicates(subset=['date'], keep='last')
        bayes_coefs_df.to_csv(f'qrisk2_{gender}_Bayesian_coefs.csv', index=False) # save the changes to the csv
        
        bayes_coefs_df['date'] = pd.to_datetime(bayes_coefs_df['date'])

        BayesianCoefsPlot(bayes_coefs_df, f'qrisk_{gender}')

    ########################################### Save Metrics #######################################
    metrics = pd.DataFrame({'Time': list(log["Accuracy"].keys()), 'Accuracy': list(log["Accuracy"].values()), 'AUROC': list(log["AUROC"].values()), 
                            'Precision': list(log["Precision"].values()), 'CalibrationSlope': list(log["CalibrationSlope"].values()), 
                            'CITL': list(log["CITL"].values()), 'OE': list(log["O/E"].values()), 'AUPRC': list(log["AUPRC"].values()),
                            'Method':list([method_str] * len(log["Accuracy"]))})

    metrics.to_csv(f'qrisk2_{gender}_performance_metrics.csv', mode='a', header=False, index=False)
    # load the data again with the new metrics included
    metrics_updated = pd.read_csv(f'qrisk2_{gender}_performance_metrics.csv')
    # remove duplicates in the Time and Method columns
    metrics_updated.drop_duplicates(subset=['Time', 'Method'], keep='last', inplace=True)
    # save the metrics csv again
    metrics_updated.to_csv(f'qrisk2_{gender}_performance_metrics.csv', mode='w', header=True, index=False)

    if 'Model Updated' in log:
        model_updates.to_csv(f'qrisk2_{gender}_model_updates.csv', mode='a', header=False, index=False)


############################################ Plot Metrics #######################################

plot_method_comparison_metrics(metrics_df = f'qrisk2_{gender}_performance_metrics.csv', recalthreshold=recalthreshold, 
                            model_updates=f'qrisk2_{gender}_model_updates.csv', model_type='qrisk2', gender=gender)

plot_count_of_patients_over_threshold_risk(threshold=0.1, model_type='qrisk2', gender=gender)
plot_calibration_yearly(model='qrisk2', method_list = method_strs, gender=gender)

plot_predictor_distributions(df, predictors=['treated_htn'], plot_type='stacked_perc', model_name=f'qrisk2_{gender}')