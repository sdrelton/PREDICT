from matplotlib import pyplot as plt
import pandas as pd
from datetime import timedelta
import numpy as np
from PREDICT import PREDICT
from PREDICT.Models import *
from PREDICT.Metrics import *
from PREDICT.Triggers import *
from PREDICT.Plots import *



def get_model_updated_log_covid(df, model, switch_times, i, model_name, undetected):
    """Get the model update log for a given model and switch time for the COVID simulation data.

    Args:
        df (pd.DataFrame): DataFrame containing the simulation data.
        model (PREDICTModel or bmb.Model): PREDICT or bayesian model to be used for prediction.
        switch_times (str): List of switch times for the model updates.
        i (int): Index for the switch time to check.
        model_name (str): Name of the model being checked for updates.
        undetected (dict): Dictionary to keep track of undetected models and their counts.

    Returns:
        int: time to detect (ttd) in days, or None if no model update detected.
    """
    mytest = PREDICT(data=df, model=model, startDate='min', endDate='max', timestep='month')
    mytest.run()
    log = mytest.getLog()
    if 'Model Updated' in log:
        dates = [date for date in log['Model Updated']]
        switch_time = pd.to_datetime(switch_times[i], dayfirst=True)
        model_update_date = next((date for date in dates if date > switch_time), None)
        if model_update_date:
            time_diff = model_update_date - switch_time
            ttd = abs(time_diff.days)
            
        else:
            ttd = None
            # add the model name to the undetected dictionary and add to the count
            count_for_model = undetected.get(model_name, 0)
            count_for_model += 1
            undetected[model_name] = count_for_model
        del mytest
        return ttd
    else:
        # if no model update, return None
        ttd = None
        del mytest
        # add the model name to the undetected dictionary and add to the count
        count_for_model = undetected.get(model_name, 0)
        count_for_model += 1
        undetected[model_name] = count_for_model
        return ttd
    
def get_model_updated_log_prev_drift(df, model, startDate, model_name, undetected):
    """Get the model update log for a given model and switch time for the sslow change, 
    outcome prevalence drift, and QRISK simulation data.

    Args:
        df (pd.DataFrame): DataFrame containing the simulation data.
        model (PREDICTModel or bmb.Model): PREDICT or bayesian model to be used for prediction.
        startDate (datetime64[ns]): Start date for when the data period begins.
        model_name (str): Name of the model being checked for updates.
        undetected (dict): Dictionary to keep track of undetected models and their counts.

    Returns:
        int: time to detect (ttd) in days, or None if no model update detected.
    """
    mytest = PREDICT(data=df, model=model, startDate='min', endDate='max', timestep='month')
    mytest.run()
    log = mytest.getLog()
    if 'Model Updated' in log:
        dates = [date for date in log['Model Updated']]
        model_update_date = next((date for date in dates if date > startDate), None)
        if model_update_date:
            time_diff = model_update_date - startDate
            ttd = abs(time_diff.days)
        else:
            ttd = None
            # add the model name to the undetected dictionary and add to the count
            count_for_model = undetected.get(model_name, 0)
            count_for_model += 1
            undetected[model_name] = count_for_model
        del mytest
        return ttd
    else:
        # if no model update, return None
        ttd = None
        del mytest
        # add the model name to the undetected dictionary and add to the count
        count_for_model = undetected.get(model_name, 0)
        count_for_model += 1
        undetected[model_name] = count_for_model
        return ttd
    
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

    # Define probabilities for selection (must sum to 1)
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

def plot_prev_over_time(df, switchDateStrings, regular_ttd, static_ttd, spc_ttd3, spc_ttd5, spc_ttd7, bayesian_ttd):
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
        # get the model update times and plot them as vertical lines
        switch_time = pd.to_datetime(switchDateStrings[-1], dayfirst=True)
        plt.vlines(x=switch_time, ymin=0, ymax=groupby_df['outcome'].max(), color='orange', linestyle='--', label='Switch Time')

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
    plt.show()


def run_recalibration_tests(df, startDate, undetected, total_runs, regular_ttd, static_ttd, spc_ttd3, spc_ttd5, spc_ttd7, recalthreshold):
    """Run recalibration tests on the given DataFrame using various triggers and return the updated undetected counts and time to detect (ttd) for each method.

    Args:
        df (pd.DataFrame): DataFrame containing the simulation data with 'date' and 'outcome' columns.
        startDate (datetime[ns]): Start date for when the data period begins.
        undetected (dict): Dictionary to keep track of undetected models and their counts.
        total_runs (int):  Total number of runs performed so far.
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
    model.trigger = TimeframeTrigger(model=model, updateTimestep=100, dataStart=df['date'].min(), dataEnd=df['date'].max())
    total_runs +=1
    ttd = get_model_updated_log_prev_drift(df, model, startDate, "Regular Testing", undetected)
    regular_ttd.append(ttd)

    ############################ Static Threshold ############################
    model = RecalibratePredictions()
    model.trigger = AUROCThreshold(model=model, prediction_threshold=recalthreshold)
    ttd = get_model_updated_log_prev_drift(df, model, startDate, "Static Threshold", undetected)
    static_ttd.append(ttd)

    ############################ SPC ############################
    model = RecalibratePredictions()
    model.trigger = SPCTrigger(model=model, input_data=df, numMonths=3, verbose=False)
    ttd = get_model_updated_log_prev_drift(df, model, startDate, "SPC3", undetected)
    spc_ttd3.append(ttd)

    model.trigger = SPCTrigger(model=model, input_data=df, numMonths=5, verbose=False)
    ttd = get_model_updated_log_prev_drift(df, model, startDate, "SPC5", undetected)
    spc_ttd5.append(ttd)

    model.trigger = SPCTrigger(model=model, input_data=df, numMonths=7, verbose=False)
    ttd = get_model_updated_log_prev_drift(df, model, startDate, "SPC7", undetected)
    spc_ttd7.append(ttd)
    return undetected, total_runs, regular_ttd, static_ttd, spc_ttd3, spc_ttd5, spc_ttd7



def run_bayes_model(undetected, bay_model, bayes_dict, df, bayesian_ttd, switchDateStrings, switchDateidx, sim_data="covid"):
    """Run the Bayesian model with a refit trigger and return the updated undetected counts, time to detect (ttd), and coefficients.

    Args:
        undetected (dict): Dictionary to keep track of undetected models and their counts.
        bay_model (bmb.Model): Bayesian model to be used for prediction.
        bayes_dict (dict): Dictionary to store Bayesian coefficients and other information.
        df (pd.DataFrame): DataFrame containing the simulation data with 'date' and 'outcome' columns.
        bayesian_ttd (list): List to store time to detect (ttd) for Bayesian model updates.
        switchDateStrings (list or datetime[ns]): List of switch dates as strings for the model updates or datetime[ns] if startDate is
                                given for non-COVID data simulation.
        switchDateidx (int): Index of the switch date to check for model updates.
        sim_data (str, optional): String of the data simulation type. Defaults to "covid".

    Returns:
        dict: Dictionary of undetected models and their counts.
        list: List of time to detect (ttd) for Bayesian model updates.
        dict: Dictionary of Bayesian coefficients and other information.
    """
    bay_model.trigger = BayesianRefitTrigger(model=bay_model, input_data=df, refitFrequency=1)
    mytest = PREDICT(data=df, model=bay_model, startDate='min', endDate='max', timestep='month')
    mytest.addLogHook(TrackBayesianCoefs(bay_model))
    mytest.run()
    log = mytest.getLog()
    if "BayesianCoefficients" in log:
        bayes_dict["BayesianCoefficients"].update(log["BayesianCoefficients"])
    if sim_data == "covid":
        ttd = get_model_updated_log_covid(df, bay_model, switchDateStrings, switchDateidx, "Bayesian", undetected)
    else:
        ttd = get_model_updated_log_prev_drift(df, bay_model, switchDateStrings, "Bayesian", undetected)
    bayesian_ttd.append(ttd)
    return undetected, bayesian_ttd, bayes_dict