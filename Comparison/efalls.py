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
from experiment_plots import *




os.environ['PYTENSOR_FLAGS'] = 'optimiser=fast_compile'


conn = pyodbc.connect(
    "DRIVER={SQL Server};"
    "SERVER=BHTS-CONNECTYO3;"
    "DATABASE=CB_2151;"
    "Trusted_Connection=yes;"
)

resultsloc = f'results/efalls'
os.makedirs(resultsloc, exist_ok=True)

# Query data from a table
query = f"SELECT * FROM tbl_final_efalls_deduped WHERE DateOnly <= '2019-01-01'"

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

# select prior twelve months used to fit scalers
prior_twelve_months = df[(df['date'] >= startDate - relativedelta(months=12)) & (df['date'] < startDate)]



# Establish connection to SQL Server
conn = pyodbc.connect(
    "DRIVER={SQL Server};"
    "SERVER=BHTS-CONNECTYO3;"
    "DATABASE=CB_2151;"
    "Trusted_Connection=yes;"
)

resultsloc = f'results/efalls'
os.makedirs(resultsloc, exist_ok=True)
os.makedirs(os.path.join(resultsloc, 'probs_and_outcomes'), exist_ok=True)
os.makedirs(os.path.join(resultsloc, 'predictor_distributions'), exist_ok=True)

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

# scale continuous variables using saved scaler parameters; require the scaler JSON to exist
""" scaler_file = os.path.join(resultsloc, 'efalls_scaler.json')
if os.path.exists(scaler_file):
    with open(scaler_file, 'r') as sf:
        scaler_params = json.load(sf)
    params = scaler_params.get('Age')
    if params is None:
        raise KeyError(f"Scaler parameters for Age not found in {scaler_file}")
    mean_val = float(params['mean'])
    scale_val = float(params['scale']) if float(params['scale']) != 0 else 1.0
    df['Age'] = (df['Age'].astype(float) - mean_val) / scale_val
else:
    raise FileNotFoundError(f"Scaler file '{scaler_file}' not found. Please run 'refit_efalls_model.py' to generate it before running this script.")
"""

# if a file to store the model update dates or performance metrics doesn't exist, then create them with the correct headers
if not os.path.exists(os.path.join(resultsloc, "efalls_performance_metrics.csv")):
    perform_metrics_df = pd.DataFrame(columns=['Time','Accuracy','AUROC','Precision','CalibrationSlope','CITL','OE','AUPRC','F1Score','Sensitivity','Specificity','CoxSnellR2','KLDivergence','Method'])
    perform_metrics_df.to_csv(os.path.join(resultsloc, "efalls_performance_metrics.csv"), index=False)

if not os.path.exists(os.path.join(resultsloc, "efalls_model_updates.csv")):
    perform_metrics_df = pd.DataFrame(columns=['date', 'method'])
    perform_metrics_df.to_csv(os.path.join(resultsloc, "efalls_model_updates.csv"), index=False)

################################## FOR SIMPLICITY RUN THE BAYESIAN METHOD SEPARATELY TO THE OTHER METHODS ##################################
method_strs = ['Baseline', 'Regular Testing', 'Static Threshold', 'SPC', 'KLD']
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

    plot_patients_per_month(df, model_type='efalls', resultsloc=resultsloc)
    
    thresh_json = os.path.join(resultsloc, f"efalls_thresh.json")
    # Initialize threshold variables
    recalthreshold = None
    citl_thresh_lower = None
    citl_thresh_upper = None
    slope_thresh_lower = None
    slope_thresh_upper = None
    
    if os.path.exists(thresh_json):
        with open(thresh_json, 'r') as jf:
            thresh_data = json.load(jf)
        # AUROC recalthreshold
        if 'auroc_recalthreshold' not in thresh_data:
            raise KeyError(f"Key 'auroc_recalthreshold' not found in {thresh_json}. Please re-run the refit script.")
        recalthreshold = float(thresh_data['auroc_recalthreshold'])
        # Other calibration thresholds
        citl_thresh_lower = float(thresh_data.get('citl_recalthreshold_lower')) if thresh_data.get('citl_recalthreshold_lower') is not None else None
        citl_thresh_upper = float(thresh_data.get('citl_recalthreshold_upper')) if thresh_data.get('citl_recalthreshold_upper') is not None else None
        slope_thresh_lower = float(thresh_data.get('slope_recalthreshold_lower')) if thresh_data.get('slope_recalthreshold_lower') is not None else None
        slope_thresh_upper = float(thresh_data.get('slope_recalthreshold_upper')) if thresh_data.get('slope_recalthreshold_upper') is not None else None
    else:
        print(f"Threshold file {thresh_json} not found. Please run 'refit_efalls_model.py' to generate it before running this script.\nSetting AUROC threshold to 0.7 for now.")
        recalthreshold = 0.7

    # expose optional thresholds in case other code wants them later
    thresh_summary = {
        'auroc': recalthreshold,
        'citl_lower': citl_thresh_lower,
        'citl_upper': citl_thresh_upper,
        'slope_lower': slope_thresh_lower,
        'slope_upper': slope_thresh_upper
    }

    # map loaded threshold names to the parameter names expected by the trigger
    lower_limit_citl = citl_thresh_lower
    upper_limit_citl = citl_thresh_upper
    lower_limit_cslope = slope_thresh_lower
    upper_limit_cslope = slope_thresh_upper

    # If the startDate and endDate are the full period then fit the initial model
    if startDate == startOfAnalysis:
        # if you want the coefficients from the model trained on the prior 6 months of data 01-06-2018 to 01-01-2019
        if not os.path.exists(os.path.join(resultsloc, "efalls_coefs.json")) and not os.path.exists(f'efalls_refitted_coefs.json'):
            raise FileNotFoundError(f"Coefficients file 'efalls_coefs.json' not found. Please run 'refit_efalls_model.py' to generate the coefficients file before running this script.")
        else:
            with open(os.path.join(resultsloc, "efalls_coefs.json"), 'r') as f:
                coefs = json.load(f)


        if not os.path.exists(os.path.join(resultsloc, f'efalls_coefs_std.json')):
            raise FileNotFoundError(f"Coefficients file 'efalls_coefs_std.json' not found. Please run 'refit_efalls_model.py' to generate the coefficients file before running this script.")
        else:
            with open(os.path.join(resultsloc, f'efalls_coefs_std.json'), 'r') as f:
                coefs_std = json.load(f)

    else: # Else will only trigger when Bayesian model is run as other methods don't require batching dates
        # select most recent coefs from the bayesian df and set them as the coefs
        bayes_coefs_df = pd.read_csv(os.path.join(resultsloc,f'efalls_Bayesian_coefs.csv'))

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
    feature_matrix = df[predictors].astype(float)
    # Dot product gives weighted sum for each row in df
    weighted_coef_sum = np.dot(feature_matrix, coef_vector)


    # Compute log-odds
    lp = coefs['Intercept'] + weighted_coef_sum

    # Estimate predictions
    curpredictions = 1 / (1 + np.exp(-lp))  # Convert to probability
    df['prediction'] = curpredictions
    
    feature_matrix_prior = prior_twelve_months[predictors].astype(float)
    weighted_coef_sum_prior = np.dot(feature_matrix_prior, coef_vector)
    lp_prior = coefs['Intercept'] + weighted_coef_sum_prior
    curpredictions_prior = 1 / (1 + np.exp(-lp_prior))
    prior_twelve_months['prediction'] = curpredictions_prior

        # clear the performance metrics and model updates for the method so we don't duplicate the date
    if (method_str != 'Bayesian') or (method_str=='Bayesian' and startDate == startOfAnalysis):
        org_metrics = pd.read_csv(os.path.join(resultsloc, 'efalls_performance_metrics.csv'))
        org_metrics = org_metrics[org_metrics["Method"] != method_str]
        org_metrics.to_csv(os.path.join(resultsloc, 'efalls_performance_metrics.csv'), mode='w', header=True, index=False)

        org_updates = pd.read_csv(os.path.join(resultsloc, 'efalls_model_updates.csv'))
        org_updates = org_updates[org_updates["method"] != method_str]
        org_updates.to_csv(os.path.join(resultsloc, 'efalls_model_updates.csv'), mode='w', header=True, index=False)

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
        mytest = PREDICT(data=df, model=model, startDate=startDate, endDate=endDate, timestep='month', recalPeriod=reg_time)


    # Static Threshold Testing
    if method_str == 'Static Threshold':
        model = RecalibratePredictions()
        #model.trigger = AUROCThreshold(model=model, update_threshold=recalthreshold)
        def custom_trigger(model, lower_limit_citl, upper_limit_citl, lower_limit_cslope, upper_limit_cslope):
            return CITLThreshold(model=model, lower_limit=lower_limit_citl, upper_limit=upper_limit_citl) or \
                CalibrationSlopeThreshold(model=model, lower_limit=lower_limit_cslope, upper_limit=upper_limit_cslope)
        model.trigger = custom_trigger(model, lower_limit_citl, upper_limit_citl, lower_limit_cslope, upper_limit_cslope)
        mytest = PREDICT(data=df, model=model, startDate=startDate, endDate=endDate, timestep='month', recalPeriod=365)


    # SPC Testing
    if method_str == 'SPC':
        model = RecalibratePredictions()
        model.trigger = SPCTrigger(model=model, input_data=df, numMonths=12, verbose=False)
        mytest = PREDICT(data=df, model=model, startDate=startDate, endDate=endDate, timestep='month', recalPeriod=365)
        
    # KL Divergence
    if method_str == 'KLD':
        model = RecalibratePredictions()
        model.trigger = KLDivergenceThreshold(model=model, initial_data=prior_twelve_months, update_threshold=0.03)
        mytest = PREDICT(data=df, model=model, startDate=startDate, endDate=endDate, timestep='month', recalPeriod=365)

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
            priors_df.to_csv(os.path.join(resultsloc, f'efalls_Bayesian_coefs.csv'), mode='w', index=False, header=True)
            
        model = BayesianModel(input_data=df, priors=priors_dict, 
                                cores=6, draws=10000, tune=3000, chains=2, verbose=False,
                                model_formula=f"outcome ~ {' + '.join(predictors)}")
                                
        model.trigger = TimeframeTrigger(model=model, updateTimestep='month', dataStart=startDate, dataEnd=endDate)
        mytest = PREDICT(data=df, model=model, startDate=startDate, endDate=endDate, timestep='month', recalPeriod=30)

    ############################################ Run Testing ###########################################

    mytest.addLogHook(Accuracy(model))
    mytest.addLogHook(AUROC(model))
    mytest.addLogHook(Precision(model))
    mytest.addLogHook(CalibrationSlope(model))
    mytest.addLogHook(CITL(model))
    mytest.addLogHook(OE(model))
    mytest.addLogHook(AUPRC(model))
    mytest.addLogHook(F1Score(model))
    mytest.addLogHook(Sensitivity(model))
    mytest.addLogHook(Specificity(model))
    mytest.addLogHook(CoxSnellR2(model))
    mytest.addLogHook(ResidualKLDivergence(model, initial_data=prior_twelve_months))
    
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

            postPredDf.to_csv(os.path.join(resultsloc, 'probs_and_outcomes', f"{method_str}_efalls_predictions_and_outcomes.csv"), mode='w', header=True, index=False)

    else: # if the model doesn't update
        noUpdatePreds = df[['date', 'outcome', 'prediction']].copy()
        noUpdatePreds.to_csv(os.path.join(resultsloc, 'probs_and_outcomes', f"{method_str}_efalls_predictions_and_outcomes.csv"), mode='w', header=True, index=False)


    if method_str == 'Bayesian':
        # if this is the first batch of dates for the bayesian model set up the coefs csv
        if startDate == startOfAnalysis:
            # create and save df with headers
            coef_columns = ['date'] + predictors + [value + 'std' for value in predictors]

        bayes_coefs_df = pd.DataFrame.from_dict(log["BayesianCoefficients"], orient='index').reset_index()
        bayes_coefs_df.columns = ['date'] + list(bayes_coefs_df.columns[1:])

        bayes_coefs_df.to_csv(os.path.join(resultsloc, f'efalls_Bayesian_coefs.csv'), mode='a', index=False, header=False)
        bayes_coefs_df = pd.read_csv(os.path.join(resultsloc, f'efalls_Bayesian_coefs.csv'))
        bayes_coefs_df = bayes_coefs_df.drop_duplicates(subset=['date'], keep='last')
        bayes_coefs_df.to_csv(os.path.join(resultsloc, f'efalls_Bayesian_coefs.csv'), index=False) # save the changes to the csv

        bayes_coefs_df['date'] = pd.to_datetime(bayes_coefs_df['date'])

        BayesianCoefsPlot(bayes_coefs_df, f'efalls', fileloc=resultsloc)

    ########################################### Save Metrics #######################################
    metrics = pd.DataFrame({'Time': list(log["Accuracy"].keys()), 'Accuracy': list(log["Accuracy"].values()), 'AUROC': list(log["AUROC"].values()), 
                            'Precision': list(log["Precision"].values()), 'CalibrationSlope': list(log["CalibrationSlope"].values()), 
                            'CITL': list(log["CITL"].values()), 'OE': list(log["O/E"].values()), 'AUPRC': list(log["AUPRC"].values()),
                            'F1Score': list(log["F1score"].values()), 'Sensitivity': list(log["Sensitivity"].values()),
                            'Specificity': list(log["Specificity"].values()), 'CoxSnellR2': list(log["CoxSnellR2"].values()), 'KLDivergence': list(log["ResidualKLDivergence"].values()),
                            'Method':list([method_str] * len(log["Accuracy"]))})

    metrics.to_csv(os.path.join(resultsloc, f'efalls_performance_metrics.csv'), mode='a', header=False, index=False)
    # load the data again with the new metrics included
    metrics_updated = pd.read_csv(os.path.join(resultsloc, f'efalls_performance_metrics.csv'))
    # remove duplicates in the Time and Method columns
    metrics_updated.drop_duplicates(subset=['Time', 'Method'], keep='last', inplace=True)
    # save the metrics csv again
    metrics_updated.to_csv(os.path.join(resultsloc, f'efalls_performance_metrics.csv'), mode='w', header=True, index=False)

    if 'Model Updated' in log:
        model_updates.to_csv(os.path.join(resultsloc, f'efalls_model_updates.csv'), mode='a', header=False, index=False)



############################################ Plot Metrics #######################################

plot_method_comparison_metrics(metrics_df = os.path.join(resultsloc, f'efalls_performance_metrics.csv'), recalthreshold=recalthreshold, 
                            model_updates=os.path.join(resultsloc, f'efalls_model_updates.csv'), model_type='efalls', fileloc=resultsloc)

plot_calibration_yearly(model='efalls', method_list = method_strs, fileloc=resultsloc)

#plot_count_of_patients_over_threshold_risk(threshold=0.1, model_type='efalls', fileloc=resultsloc)

plot_predictor_distributions(df, predictors=['Falls'], plot_type='stacked_perc', model_name='efalls', fileloc=resultsloc)