# try to create a model using the terms from connect bradford refit
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
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings('ignore')
import os
import json
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import binom



os.environ['PYTENSOR_FLAGS'] = 'optimiser=fast_compile'

# Establish connection to SQL Server
conn = pyodbc.connect(
    "DRIVER={SQL Server};"
    "SERVER=BHTS-CONNECTYO3;"
    "DATABASE=CB_2151;"
    "Trusted_Connection=yes;"
)

# Query data from a table
query = f"SELECT * FROM tbl_final_efalls_deduped WHERE DateOnly >= '2019-01-01'"

# Store query result in a dataframe
df = pd.read_sql(query, conn)

# Close the connection
conn.close()

# apply transformation to polypharmacy column
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


df = df[["Fall_Outcome", "DateOnly"]+predictors]

df.rename(columns={"DateOnly": "date", "Fall_Outcome":"outcome"}, inplace=True)
df['date'] = pd.to_datetime(df['date'])
df["Female"] = df["Female"].astype(int)
df["Current_Smoker"] = df["Current_Smoker"].astype(int)

# scale continuous variables:
scaler = StandardScaler()

scaled_age = scaler.fit_transform(df[['Age']])
df['Age'] = pd.DataFrame(scaled_age, columns=['Age'])

# # TODO: Do we want to scale Polypharmacy too?
# scaled_Polypharmacy = scaler.fit_transform(df[['Polypharmacy']])
# df['Polypharmacy'] = pd.DataFrame(scaled_Polypharmacy, columns=['Polypharmacy'])

# if a file to store the model update dates or performance metrics doesn't exist, then create them with the correct headers
if not os.path.exists("efalls_performance_metrics.csv"):
    perform_metrics_df = pd.DataFrame(columns=['Time','Accuracy','AUROC','Precision','CalibrationSlope','CITL','OE','AUPRC','Method'])
    perform_metrics_df.to_csv("efalls_performance_metrics.csv", index=False)

if not os.path.exists("efalls_model_updates.csv"):
    perform_metrics_df = pd.DataFrame(columns=['date', 'method'])
    perform_metrics_df.to_csv("efalls_model_updates.csv", index=False)

################################## FOR SIMPLICITY RUN THE BAYESIAN METHOD SEPARATELY TO THE OTHER METHODS ##################################
method_strs = ['Baseline', 'Regular Testing', 'Static Threshold', 'SPC']
#method_strs = ['Bayesian']

for method_str in method_strs:

    print(f"Running efalls Model using {method_str} method...")
    startOfAnalysis = pd.to_datetime('01-01-2019', dayfirst=True)

    # Only change these dates when batching for the Bayesian method
    startDate = startOfAnalysis #pd.to_datetime('01-01-2019', dayfirst=True)
    #startDate = pd.to_datetime('01-01-2019', dayfirst=True)
    #endDate = pd.to_datetime('01-05-2019', dayfirst=True)
    endDate = pd.to_datetime('19-08-2024', dayfirst=True) # Most recent record minus 1 year: '2025-08-19'

    df = df[df['date']<= endDate]

    plot_patients_per_month(df, model_type='efalls')

    if not os.path.exists(f"efalls_auroc_thresh.txt"):
        print(f"Threshold file 'efalls_auroc_thresh.txt' not found. Please run 'refit_efalls_model.py' to generate the threshold file or manually create the threshold file before running this script.\nSetting threshold to 0.7 for now.")
        recalthreshold = 0.7
    else:
        with open(f"efalls_auroc_thresh.txt", "r") as file:
            recalthreshold = float(file.read())

    # If the startDate and endDate are the full period then fit the initial model
    if startDate == startOfAnalysis:
        # if you want the coefficients from the model trained on the prior 6 months of data 01-06-2018 to 01-01-2019
        with open(f'efalls_coefs.json', 'r') as f:
            coefs = json.load(f)

        # # if you want to use the refitted coefficients from efalls paper
        # with open(f'efalls_refitted_coefs.json', 'r') as f:
        #     coefs = json.load(f)


        coefs_std = {key: 0.25 for key in coefs}

    else: # Else will only trigger when Bayesian model is run as other methods don't require batching dates
        # select most recent coefs from the bayesian df and set them as the coefs
        bayes_coefs_df = pd.read_csv(f'efalls_Bayesian_coefs.csv')
        
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
    coef_vector = np.array([coefs[f] for f in predictors])    
    feature_matrix = np.column_stack([df[f] for f in predictors])
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
        org_metrics = pd.read_csv(f'efalls_performance_metrics.csv')
        org_metrics = org_metrics[org_metrics["Method"] != method_str]
        org_metrics.to_csv(f'efalls_performance_metrics.csv', mode='w', header=True, index=False)

        org_updates = pd.read_csv(f'efalls_model_updates.csv')
        org_updates = org_updates[org_updates["method"] != method_str]
        org_updates.to_csv(f'efalls_model_updates.csv', mode='w', header=True, index=False)

    # Baseline Testing
    if method_str == 'Baseline':
        model = RecalibratePredictions()
        model.trigger = TimeframeTrigger(model=model, updateTimestep=9_999, dataStart=startDate, dataEnd=endDate)
        mytest = PREDICT(data=df, model=model, startDate=startDate, endDate=endDate, timestep='month', model_name=f'{method_str}_efalls')

    # Regular Testing
    if method_str == 'Regular Testing':
        reg_time = 3*365
        model = RecalibratePredictions()
        model.trigger = TimeframeTrigger(model=model, updateTimestep=reg_time, dataStart=startDate, dataEnd=endDate)
        mytest = PREDICT(data=df, model=model, startDate=startDate, endDate=endDate, timestep='month', recal_period=reg_time, model_name=f'{method_str}_efalls')


    # Static Threshold Testing
    if method_str == 'Static Threshold':
        model = RecalibratePredictions()
        model.trigger = AUROCThreshold(model=model, update_threshold=recalthreshold)
        mytest = PREDICT(data=df, model=model, startDate=startDate, endDate=endDate, timestep='month', recal_period=365, model_name=f'{method_str}_efalls')


    # SPC Testing
    if method_str == 'SPC':
        model = RecalibratePredictions()
        model.trigger = SPCTrigger(model=model, input_data=df, numMonths=12, verbose=False)
        mytest = PREDICT(data=df, model=model, startDate=startDate, endDate=endDate, timestep='month', recal_period=365, model_name=f'{method_str}_efalls')


    # Bayesian Testing
    if method_str == 'Bayesian':
        priors_dict = {"Intercept": (coefs["Intercept"], coefs_std["Intercept"])}
        for predictor in predictors:
            priors_dict[predictor] = (coefs[predictor], coefs_std[predictor])

        # add the initial coefficients to the csv file to plot Bayesian coefficients over time
        if startDate == startOfAnalysis:
            priors_df = pd.DataFrame([priors_dict])
            # add the date
            priors_df.insert(0, 'date', startDate)
            priors_df.to_csv(f'efalls_Bayesian_coefs.csv', mode='w', index=False, header=True)
            
        model = BayesianModel(input_data=df, priors=priors_dict, 
                                cores=1, draws=100, tune=100, chains=2, verbose=False,
                                model_formula=f"outcome ~ {' + '.join(predictors)}")
                                
        model.trigger = TimeframeTrigger(model=model, updateTimestep='month', dataStart=startDate, dataEnd=endDate)
        mytest = PREDICT(data=df, model=model, startDate=startDate, endDate=endDate, timestep='month', recal_period=30, model_name=f'{method_str}_efalls', startOfAnalysis=startOfAnalysis)

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

            postPredDf.to_csv(f"probs_and_outcomes/{method_str}_efalls_predictions_and_outcomes.csv", mode='w', header=True, index=False)

    else: # if the model doesn't update
        noUpdatePreds = df[['date', 'outcome', 'prediction']].copy()
        noUpdatePreds.to_csv(f"probs_and_outcomes/{method_str}_efalls_predictions_and_outcomes.csv", mode='w', header=True, index=False)


    if method_str == 'Bayesian':
        # if this is the first batch of dates for the bayesian model set up the coefs csv
        if startDate == startOfAnalysis:
            # create and save df with headers
            coef_columns = ['date'] + predictors + [value + 'std' for value in predictors]

        bayes_coefs_df = pd.DataFrame.from_dict(log["BayesianCoefficients"], orient='index').reset_index()
        bayes_coefs_df.columns = ['date'] + list(bayes_coefs_df.columns[1:])
        
        bayes_coefs_df.to_csv(f'efalls_Bayesian_coefs.csv', mode='a', index=False, header=False)
        bayes_coefs_df = pd.read_csv(f'efalls_Bayesian_coefs.csv')
        bayes_coefs_df = bayes_coefs_df.drop_duplicates(subset=['date'], keep='last')
        bayes_coefs_df.to_csv(f'efalls_Bayesian_coefs.csv', index=False) # save the changes to the csv
        
        bayes_coefs_df['date'] = pd.to_datetime(bayes_coefs_df['date'])

        BayesianCoefsPlot(bayes_coefs_df, f'efalls')

    ########################################### Save Metrics #######################################
    metrics = pd.DataFrame({'Time': list(log["Accuracy"].keys()), 'Accuracy': list(log["Accuracy"].values()), 'AUROC': list(log["AUROC"].values()), 
                            'Precision': list(log["Precision"].values()), 'CalibrationSlope': list(log["CalibrationSlope"].values()), 
                            'CITL': list(log["CITL"].values()), 'OE': list(log["O/E"].values()), 'AUPRC': list(log["AUPRC"].values()),
                            'Method':list([method_str] * len(log["Accuracy"]))})

    metrics.to_csv(f'efalls_performance_metrics.csv', mode='a', header=False, index=False)
    # load the data again with the new metrics included
    metrics_updated = pd.read_csv(f'efalls_performance_metrics.csv')
    # remove duplicates in the Time and Method columns
    metrics_updated.drop_duplicates(subset=['Time', 'Method'], keep='last', inplace=True)
    # save the metrics csv again
    metrics_updated.to_csv(f'efalls_performance_metrics.csv', mode='w', header=True, index=False)

    if 'Model Updated' in log:
        model_updates.to_csv(f'efalls_model_updates.csv', mode='a', header=False, index=False)

    


############################################ Plot Metrics #######################################

plot_method_comparison_metrics(metrics_df = f'efalls_performance_metrics.csv', recalthreshold=recalthreshold, 
                            model_updates=f'efalls_model_updates.csv', model_type='efalls')

plot_calibration_yearly(model='efalls', method_list = method_strs)

#plot_count_of_patients_over_threshold_risk(threshold=0.1, model_type='efalls')

plot_predictor_distributions(df, predictors=['Falls'], plot_type='stacked_perc', model_name='efalls')