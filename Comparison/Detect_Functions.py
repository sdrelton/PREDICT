from matplotlib import pyplot as plt
import pandas as pd
from datetime import timedelta
import numpy as np
from PREDICT import PREDICT
from PREDICT.Models import *
from PREDICT.Metrics import *
from PREDICT.Triggers import *
from PREDICT.Plots import *
from dateutil.relativedelta import relativedelta
import logging



def get_model_updated_log(df, model, model_name, undetected, detectDate):
    """Get the model update log for a given model and simulation type.

    Args:
        df (pd.DataFrame): DataFrame containing the simulation data.
        model (PREDICTModel or bmb.Model): PREDICT or bayesian model to be used for prediction.
        model_name (str): Name of the model being checked for updates.
        undetected (dict): Dictionary to keep track of undetected models and their counts.
        detectDate (datetime64[ns], optional): Date when the model is either deployed (non-COVID) or when the switch date is given (COVID).

    Returns:
        int: Time to detect (ttd) in days, or None if no model update detected.
    """
    if model_name == "Regular Testing": # if regular testing then recalibrate using the last 3 years 
        mytest = PREDICT(data=df, model=model, startDate='min', endDate='max', timestep='month', recal_period=3*365)
    else:
        mytest = PREDICT(data=df, model=model, startDate='min', endDate='max', timestep='month')
    mytest.run()
    log = mytest.getLog()
    
    if 'Model Updated' in log:
        dates = log['Model Updated']        
        model_update_date = next((date for date in dates if date > detectDate), None)
        
        if model_update_date:
            ttd = abs((model_update_date - detectDate).days)
        else:
            ttd = None
            undetected[model_name] = undetected.get(model_name, 0) + 1
        
        del mytest
        return ttd
    
    # If no model update, add to dictionary of methods that don't detect drift
    undetected[model_name] = undetected.get(model_name, 0) + 1
    del mytest
    return None
    
def get_binom_from_normal(mean, std, num_patients, threshold):
    """Get a binomial distribution for an outcome being over a certain threshold,
    using the mean and std of a normal distribution.

    Args:
        mean (float): Average value of the normal distribution.
        std (float): Standard deviation of the normal distribution.
        num_patients (int): Number of patients to simulate.
        threshold (float): Value which determines whether outcome or positive.

    Returns:
        np.array: Binomial distribution of outcomes.
    """
    # Generate normal distribution
    normal_dist = np.random.normal(mean, std, num_patients)
    
    # Calculate the percentage of patients above the threshold
    perc_over_threshold = np.sum(normal_dist >= threshold) / num_patients
    
    # Generate binomial distribution based on the percentage
    binom_dist = np.random.binomial(1, perc_over_threshold, num_patients)
    
    return binom_dist

def select_ethnic_group(num_patients):
    """ Randomly assign each patient to an ethnic group based on predefined probabilities.

    Args:
        num_patients (int): Number of patients to select an ethnic group for.

    Returns:
        dict: Dictionary of ethnic groups with binary assignments for each patient.
    """
    # Define ethnicity groups
    ethnic_groups = ["White", "Indian", "Pakistani", "Bangladeshi", "Other_Asian",
                    "Black_Caribbean", "Black_African", "Chinese", "Other"]

    # Define probabilities for selection (must sum to 1) probabilities from Hippisley-Cox et al., 2008
    probabilities = [0.973, 0.0046, 0.0025, 0.0025, 0.0015, 0.0050, 0.0047, 0.0015, 0.0047]

    # Initialize dictionary with empty lists
    ethnicity_assignments = {eth: [] for eth in ethnic_groups}

    # Repeat selection num_patients times
    for _ in range(num_patients):
        selected_ethnicity = np.random.choice(ethnic_groups, p=probabilities)

        # Assign selected ethnicity as "1", others as "0"
        for eth in ethnic_groups:
            ethnicity_assignments[eth].append(1 if eth == selected_ethnicity else 0)
    
    ethnicity_assignments = {eth: np.array(vals) for eth, vals in ethnicity_assignments.items()}

    return ethnicity_assignments

def plot_prev_over_time(df, switchDateStrings, regular_ttd, static_ttd, spc_ttd3, spc_ttd5, spc_ttd7, bayesian_ttd, sim_data=None):
    """Plot the prevalence of an outcome over time, with vertical lines indicating model update times.

    Args:
        df (pd.DataFrame): DataFrame containing the simulation data with 'date' and 'outcome' columns.
        switchDateStrings (list or None): List of switch dates as strings, or None if not applicable.
        regular_ttd (list): List of time to detect (ttd) for regular testing model updates.
        static_ttd (list): List of time to detect (ttd) for static threshold model updates.
        spc_ttd3 (list): List of time to detect (ttd) for SPC 3 months model updates.
        spc_ttd5 (list): List of time to detect (ttd) for SPC 5 months model updates.
        spc_ttd7 (list): List of time to detect (ttd) for SPC 7 months model updates.
        bayesian_ttd (list): List of time to detect (ttd) for Bayesian model updates.
    """

    # If we want to plot a different simulated data prevalence:
    # save times and grouped prevalence in a df to plot at the end? - another column to say which number run it is for new lines
    plt.figure(figsize=(10, 5)) # plot prevalence over time for each switchTime - start with just the final one first

    # groupby the date and get the sum of the outcome
    groupby_df = df.groupby('date').agg({'outcome': 'sum'}).reset_index()

    plt.plot(groupby_df['date'], groupby_df['outcome'], label='Prevalence', color='blue')

    if switchDateStrings is not None:
        switch_time = pd.to_datetime(switchDateStrings[-1], dayfirst=True)
        plt.vlines(x=switch_time, ymin=0, ymax=groupby_df['outcome'].max(), color='orange', linestyle='--', label='Switch Time')
    else:
        switch_time = df['date'].min()  # Use the minimum date in the DataFrame if no switch date is provided

    if len(regular_ttd) > 0 and regular_ttd[-1] is not None:
        regular_update = switch_time + timedelta(days=regular_ttd[-1])
        plt.vlines(x=regular_update, ymin=0, ymax=groupby_df['outcome'].max(), color='black', linestyle='--', label='Regular Testing Model Update Time', alpha=0.6)
    if len(static_ttd) > 0 and static_ttd[-1] is not None:
        static_update = switch_time + timedelta(days=static_ttd[-1])
        plt.vlines(x=static_update, ymin=0, ymax=groupby_df['outcome'].max(), color='purple', linestyle='--', label='Static Threshold Model Update Time', alpha=0.6)
    if len(spc_ttd3) > 0 and spc_ttd3[-1] is not None: 
        spc_update3 = switch_time + timedelta(days=spc_ttd3[-1])
        plt.vlines(x=spc_update3, ymin=0, ymax=groupby_df['outcome'].max(), color='green', linestyle='--', label='SPC 3 months Model Update Time', alpha=0.6)
    if len(spc_ttd5) > 0 and spc_ttd5[-1] is not None:
        spc_update5 = switch_time + timedelta(days=spc_ttd5[-1])
        plt.vlines(x=spc_update5, ymin=0, ymax=groupby_df['outcome'].max(), color='pink',  linestyle='--', label='SPC 5 months Model Update Time', alpha=0.6)
    if len(spc_ttd7) > 0 and spc_ttd7[-1] is not None:
        spc_update7 = switch_time + timedelta(days=spc_ttd7[-1])
        plt.vlines(x=spc_update7, ymin=0, ymax=groupby_df['outcome'].max(), color='grey', linestyle='--', label='SPC 7 months Model Update Time', alpha=0.6)
    if len(bayesian_ttd) > 0 and bayesian_ttd[-1] is not None:
        bayesian_update = switch_time + timedelta(days=bayesian_ttd[-1])
        plt.vlines(x=bayesian_update, ymin=0, ymax=groupby_df['outcome'].max(), linestyle='-', label='Bayesian Model Update Time')

    plt.xlabel("Date")
    plt.ylabel("Prevalence")
    plt.legend()
    # save figure
    plt.savefig(f"../docs/images/monitoring/prev_over_time/prevalence_over_time_{sim_data}.png", dpi=600, bbox_inches='tight')
    plt.show()

    


def run_recalibration_tests(df, detectDate, undetected, regular_ttd, static_ttd, spc_ttd3, spc_ttd5, spc_ttd7, recalthreshold):
    """Run recalibration tests on the given DataFrame using various triggers and return the updated undetected counts and time to detect (ttd) for each method.

    Args:
        df (pd.DataFrame): DataFrame containing the simulation data with 'date' and 'outcome' columns.
        detectDate (datetime64[ns]): Date when the model is either deployed (non-COVID) or when the switch date is given (COVID).
        undetected (dict): Dictionary to keep track of undetected models and their counts.
        regular_ttd (list): List of time to detect (ttd) for regular testing model updates.
        static_ttd (list): List of time to detect (ttd) for static threshold model updates.
        spc_ttd3 (list): List of time to detect (ttd) for SPC 3 months model updates.
        spc_ttd5 (list): List of time to detect (ttd) for SPC 5 months model updates.
        spc_ttd7 (list): List of time to detect (ttd) for SPC 7 months model updates.
        recalthreshold (float): Threshold for the static threshold recalibration method.

    Returns:
        dict: Dictionary of undetected models and their counts.
        int: Total number of runs performed.
        list: List of time to detect (ttd) for regular testing model updates.
        list: List of time to detect (ttd) for static threshold model updates.
        list: List of time to detect (ttd) for SPC 3 months model updates.
        list: List of time to detect (ttd) for SPC 5 months model updates.
        list: List of time to detect (ttd) for SPC 7 months model updates.
    """
    ########################## Regular Testing ##########################
    model = RecalibratePredictions()
    model.trigger = TimeframeTrigger(model=model, updateTimestep=182, dataStart=df['date'].min(), dataEnd=df['date'].max())
    ttd = get_model_updated_log(df, model, model_name="Regular Testing", undetected=undetected, detectDate=detectDate)
    regular_ttd.append(ttd)

    ############################ Static Threshold ############################
    model = RecalibratePredictions()
    model.trigger = AUROCThreshold(model=model, update_threshold=recalthreshold)
    ttd = get_model_updated_log(df, model, model_name="Static Threshold", undetected=undetected, detectDate=detectDate)
    static_ttd.append(ttd)

    ############################ SPC ############################
    model = RecalibratePredictions()
    model.trigger = SPCTrigger(model=model, input_data=df, numMonths=3, verbose=False)
    ttd = get_model_updated_log(df, model, model_name="SPC3", undetected=undetected, detectDate=detectDate)
    
    spc_ttd3.append(ttd)

    model.trigger = SPCTrigger(model=model, input_data=df, numMonths=5, verbose=False)
    ttd = get_model_updated_log(df, model, model_name="SPC5", undetected=undetected, detectDate=detectDate)
    spc_ttd5.append(ttd)

    model.trigger = SPCTrigger(model=model, input_data=df, numMonths=7, verbose=False)
    ttd = get_model_updated_log(df, model, model_name="SPC7", undetected=undetected, detectDate=detectDate)
    spc_ttd7.append(ttd)
    return undetected, regular_ttd, static_ttd, spc_ttd3, spc_ttd5, spc_ttd7


def find_bayes_coef_change(bayesian_coefficients, detectDate, undetected, model_name="Bayesian", threshold=0.1):
    """Find the first significant change in Bayesian coefficients after a given detection date. 
    Work out the time to detect (ttd) from the first significant change in coefficients.

    Args:
        bayesian_coefficients (dict): Dictionary containing Bayesian coefficients with timestamps as keys.
        detectDate (datetime64[ns]): Date when the model is either deployed (non-COVID) or when the switch date is given (COVID).
        bayes_dict (dict): Dictionary to store Bayesian coefficients and other information.
        undetected (dict): Dictionary to keep track of undetected models and their counts.
        model_name (str): Name of the method being used. Defaults to "Bayesian".
        threshold (float): Threshold for significant coefficient change. Defaults to 0.1.

    Returns:
        int: Number of days to detect drift in the model.
    """
    significant_timestamps = []

    timestamps = sorted(bayesian_coefficients.keys()) 

    for i in range(len(timestamps) - 1):
        curr_timestamp = timestamps[i]
        next_timestamp = timestamps[i + 1]

        curr_coeffs = bayesian_coefficients[curr_timestamp]
        next_coeffs = bayesian_coefficients[next_timestamp]

        for key in curr_coeffs:
            curr_value = curr_coeffs[key][0]  # Get coefficient value
            next_value = next_coeffs[key][0]

            if abs(next_value - curr_value) > abs(curr_value) * threshold:  # More than X% difference
                significant_timestamps.append(next_timestamp)
                print(f"Significant change detected in coefficient '{key}' from {curr_value} to {next_value} at timestamp {next_timestamp}")
                break  # Move to the next timestamp after finding a change in a coefficient at that timestamp
    
    # Assuming the first significant increase after the start time or switch time is the one we want to calculate time to detect from
    filtered_timestamps = [ts for ts in significant_timestamps if ts >= detectDate]
    if len(filtered_timestamps) > 0:
        ttd = (abs((detectDate) - filtered_timestamps[0]).days)
    else:
        ttd = None
        undetected[model_name] = undetected.get(model_name, 0) + 1

    return ttd



def run_bayes_model(undetected, bay_model, bayes_dict, df, bayesian_ttd, detectDate):
    """Run the Bayesian model with a refit trigger and return the updated undetected counts, time to detect (ttd), and coefficients.

    Args:
        undetected (dict): Dictionary to keep track of undetected models and their counts.
        bay_model (bmb.Model): Bayesian model to be used for prediction.
        bayes_dict (dict): Dictionary to store Bayesian coefficients and other information.
        df (pd.DataFrame): DataFrame containing the simulation data with 'date' and 'outcome' columns.
        bayesian_ttd (list): List to store time to detect (ttd) for Bayesian model updates.
        detectDate (datetime64[ns]): Date when the model is either deployed (non-COVID) or when the switch date is given (COVID).

    Returns:
        dict: Dictionary of undetected models and their counts.
        list: List of time to detect (ttd) for Bayesian model updates.
        dict: Dictionary of Bayesian coefficients and other information.
    """
    bay_model.trigger = BayesianRefitTrigger(model=bay_model, input_data=df, refitFrequency=1)
    mytest = PREDICT(data=df, model=bay_model, startDate='min', endDate='max', timestep='month', recal_period=30)
    mytest.addLogHook(TrackBayesianCoefs(bay_model))
    mytest.run()
    log = mytest.getLog()
    if "BayesianCoefficients" in log:
        bayes_dict["BayesianCoefficients"].update(log["BayesianCoefficients"])
    #ttd = get_model_updated_log(df, bay_model, model_name="Bayesian", undetected=undetected, detectDate=detectDate)
    ttd = find_bayes_coef_change(bayes_dict["BayesianCoefficients"], detectDate=detectDate, undetected=undetected, threshold=0.1)
    bayesian_ttd.append(ttd)
    return undetected, bayesian_ttd, bayes_dict




def get_metrics_recal_methods(df, custom_impact, recalthreshold):
    """Get metrics for different recalibration methods on the given DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the simulation data with 'date' and 'outcome' columns.
        custom_impact (float): Either the custom impact on the outcome or the prevalence of a condition.
        recalthreshold (float): Threshold for the static threshold recalibration method.

    Returns:
        pd.DataFrame: DataFrame containing the metrics for each recalibration method, including accuracy, AUROC, precision, and impact or prevalence.
    """
    metrics = []

    # Regular Testing
    model = RecalibratePredictions()
    model.trigger = TimeframeTrigger(model=model, updateTimestep=182, dataStart=df['date'].min(), dataEnd=df['date'].max())
    mytest = PREDICT(data=df, model=model, startDate='min', endDate='max', timestep='month')
    mytest.addLogHook(Accuracy(model))
    mytest.addLogHook(AUROC(model))
    mytest.addLogHook(Precision(model))
    mytest.addLogHook(CalibrationSlope(model))
    mytest.addLogHook(CITL(model))
    mytest.addLogHook(OE(model))
    mytest.addLogHook(AUPRC(model))
    mytest.run()
    log = mytest.getLog()

    metrics.append(pd.DataFrame({'Time': list(log["Accuracy"].keys()), 
                                'Accuracy': list(log["Accuracy"].values()), 
                                'AUROC': list(log["AUROC"].values()), 
                                'Precision': list(log["Precision"].values()),
                                'CalibrationSlope': list(log["CalibrationSlope"].values()), 
                                'CITL': list(log["CITL"].values()),
                                'OE': list(log["O/E"].values()),
                                'AUPRC': list(log["AUPRC"].values()),
                                'impact_or_prev': [str(custom_impact)] * len(log["Accuracy"]), 
                                'Method': ['Regular Testing'] * len(log["Accuracy"])}))

    # Static Threshold Testing
    model = RecalibratePredictions()
    model.trigger = AUROCThreshold(model=model, update_threshold=recalthreshold)
    mytest = PREDICT(data=df, model=model, startDate='min', endDate='max', timestep='month')
    mytest.addLogHook(Accuracy(model))
    mytest.addLogHook(AUROC(model))
    mytest.addLogHook(Precision(model))
    mytest.addLogHook(CalibrationSlope(model))
    mytest.addLogHook(CITL(model))
    mytest.addLogHook(OE(model))
    mytest.addLogHook(AUPRC(model))
    mytest.run()
    log = mytest.getLog()

    metrics.append(pd.DataFrame({'Time': list(log["Accuracy"].keys()), 
                                'Accuracy': list(log["Accuracy"].values()), 
                                'AUROC': list(log["AUROC"].values()), 
                                'Precision': list(log["Precision"].values()), 
                                'CalibrationSlope': list(log["CalibrationSlope"].values()),
                                'CITL': list(log["CITL"].values()),
                                'OE': list(log["O/E"].values()),
                                'AUPRC': list(log["AUPRC"].values()), 
                                'impact_or_prev': [str(custom_impact)] * len(log["Accuracy"]), 
                                'Method': ['Static Threshold'] * len(log["Accuracy"])}))

    # SPC Testing (3, 5, 7 months)
    for numMonths in [3, 5, 7]:
        model = RecalibratePredictions()
        model.trigger = SPCTrigger(model=model, input_data=df, numMonths=numMonths, verbose=False)
        mytest = PREDICT(data=df, model=model, startDate='min', endDate='max', timestep='month')
        mytest.addLogHook(Accuracy(model))
        mytest.addLogHook(AUROC(model))
        mytest.addLogHook(Precision(model))
        mytest.addLogHook(CalibrationSlope(model))
        mytest.addLogHook(CITL(model))
        mytest.addLogHook(OE(model))
        mytest.addLogHook(AUPRC(model))
        mytest.run()
        log = mytest.getLog()

        metrics.append(pd.DataFrame({'Time': list(log["Accuracy"].keys()), 
                                    'Accuracy': list(log["Accuracy"].values()), 
                                    'AUROC': list(log["AUROC"].values()), 
                                    'Precision': list(log["Precision"].values()), 
                                    'CalibrationSlope': list(log["CalibrationSlope"].values()), 
                                    'CITL': list(log["CITL"].values()),
                                    'OE': list(log["O/E"].values()),
                                    'AUPRC': list(log["AUPRC"].values()),
                                    'impact_or_prev': [str(custom_impact)] * len(log["Accuracy"]), 
                                    'Method': [f'SPC{numMonths}'] * len(log["Accuracy"])}))

    return pd.concat(metrics, ignore_index=True)

def prevent_constant_variable(df, startDate, endDate, ):
    """
    Prevents constant variables in the dataframe (e.g., if no one has AF) by checking each month of data and flipping a value if a column is constant.
    Args:
        df (pd.DataFrame): The input dataframe containing patient data.
        startDate (datetime): The start date of the data.
        endDate (datetime): The end date of the data.
    Returns:
        pd.DataFrame: The modified dataframe with constant variables handled."""

    # for every month of data in df check to make sure none of the columns are consistently one value
    currentWindowStart = startDate
    timestep = relativedelta(months=1)
    currentWindowEnd = startDate + timestep
    while currentWindowEnd <= endDate:
        # filter the dataframe to only include data within the current window
        df_window = df[(df['date'] >= currentWindowStart) & (df['date'] < currentWindowEnd)]

        # if any of the diseases are constant (e.g. all 0s or all 1s), flip a value in the dataframe
        for col in ['Family_CHD', 'Current_smoker', 'Treated_HTN', 'DM', 'RA', 'AF', 'Renal_disease']:
            if df_window[col].nunique() == 1:
                logging.warning(f"Warning: '{col}' has no assigned patients between {currentWindowStart} and {currentWindowEnd}. Forcing one assignment to prevent constant error.")
                # copy a random row of data and flip the value of the disease
                random_idx = np.random.choice(df_window.index)
                # add new rows to original dataframe
                df = pd.concat([df, df_window.loc[[random_idx]].copy()], ignore_index=False)
                df.loc[df.index[-1], col] = 1 - df[col].iloc[-1]
        
        # if any of the ethnicities are constant (e.g. all 0s or all 1s), switch the 1 to to the column that is constant
        ethnicity_cols = ['White', 'Indian', 'Pakistani', 'Bangladeshi', 'Other_Asian',
                    'Black_Caribbean', 'Black_African', 'Chinese', 'Other']
        for col in ethnicity_cols:
            if df_window[col].nunique()==1:
                #print(f"Warning: '{col}' has no assigned patients between {currentWindowStart} and {currentWindowEnd}. Forcing one assignment to prevent constant error.")
                random_idx = np.random.choice(df_window.index)
                df = pd.concat([df, df_window.loc[[random_idx]].copy()], ignore_index=False)
                df.loc[df.index[-1], ethnicity_cols] = 0  # Clear previous one-hot encoding

                # Assign the current ethnicity to the newly added row
                df.loc[df.index[-1], col] = 1

        
        # move to the next window
        currentWindowStart += timestep
        currentWindowEnd += timestep


    df.reset_index(drop=True, inplace=True)
    return df

def plot_time_to_detect(csv_name, str_name):
    """Plot a figure showing the time to detect change in outcomes for different impacts.

    Args:
        csv_name (str): File name of the CSV containing time to detect data.
        str_name (str): Name of string to add to image filename e.g., "fast_change" or "slow_change".
    """
    df = pd.read_csv(csv_name)
    plt.figure(figsize=(10, 5))
    plt.title(f"Time to Detect Change in Outcomes")
    plt.plot('impact', 'regular_ttd', color='#f781bf', label='Regular Testing', alpha=0.6, linewidth=1, data=df)
    plt.plot('impact', 'static_ttd', color='#984ea3', label='Static Threshold', alpha=0.6, linewidth=1, data=df)
    plt.plot('impact', 'spc_ttd3', color='#4daf4a', label='SPC 3 months', alpha=0.6, linewidth=1, data=df)
    plt.plot('impact', 'spc_ttd5', color='#377eb8', label='SPC 5 months', alpha=0.6, linewidth=1, data=df)
    plt.plot('impact', 'spc_ttd7', color='#ff7f00', label='SPC 7 months', alpha=0.6, linewidth=1, data=df)
    plt.plot('impact', 'bayesian_ttd', color='#a65628', label='Bayesian', alpha=0.6, linewidth=1, data=df)
    plt.xlabel("Impact Increase Size")
    plt.ylabel("Time to Detect (days)")
    plt.savefig(f"../docs/images/monitoring/time_to_detect_change_{str_name}.png", dpi=600, bbox_inches='tight')
    plt.show()


def update_ttd_table(regular_ttd, static_ttd, spc_ttd3, spc_ttd5, spc_ttd7, bayesian_ttd, custom_impact, ttd_csv_file):
    """
    Update the time-to-detect (TTD) table with new values and generate plots.
    
    Args:
        regular_ttd (list): List containing time to detect for regular testing.
        static_ttd (list): List containing time to detect for static threshold.
        spc_ttd3 (list): List containing time to detect for SPC with window size 3 months.
        spc_ttd5 (list): List containing time to detect for SPC with window size 5 months.
        spc_ttd7 (list): List containing time to detect for SPC with window size 7 months.
        bayesian_ttd (list): List containing time to detect for Bayesian method.
        custom_impact (float): The custom impact/prevalence value used in the simulation.
        ttd_csv_file (str): Path to the CSV file where TTD data is stored.
    """ 
    
    
    # Load ttd dataframe
    ttd_df = pd.read_csv(ttd_csv_file)

    # Update time to detect values
    if regular_ttd and regular_ttd[0] is not None:
        ttd1 = int(regular_ttd[0])
    else:
        print("Regular testing did not detect any change.")
        ttd1 = ''

    if static_ttd and static_ttd[0] is not None:
        ttd2 = int(static_ttd[0])
    else:
        print("Static threshold did not detect any change.")
        ttd2 = ''

    if spc_ttd3 and spc_ttd3[0] is not None:
        ttd3 = int(spc_ttd3[0])
    else:
        print("SPC3 did not detect any change.")
        ttd3 = ''

    if spc_ttd5 and spc_ttd5[0] is not None:
        ttd4 = int(spc_ttd5[0])
    else:
        print("SPC5 did not detect any change.")
        ttd4 = ''

    if spc_ttd7 and spc_ttd7[0] is not None:
        ttd5 = int(spc_ttd7[0])
    else:
        print("SPC7 did not detect any change.")
        ttd5=''
        
    if bayesian_ttd and bayesian_ttd[0] is not None:
        ttd7 = int(bayesian_ttd[0])
    else:
        print("Bayesian did not detect any change.")
        ttd7 = ''
    impact_val = custom_impact

    ttd_df.loc[len(ttd_df)] = [ttd1, ttd2, ttd3, ttd4, ttd5, ttd7, impact_val]

    # Save updated data
    ttd_df.to_csv(ttd_csv_file, index=False)


