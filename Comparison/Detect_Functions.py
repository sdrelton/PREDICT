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
    # If we want to plot a different simulated data prevalence:
    # save times and grouped prevalence in a df to plot at the end? - another column to say which number run it is for new lines

    # plot prevalence over time for each switchTime - start with just the final one first
    plt.figure(figsize=(10, 5))

    # groupby the date and get the sum of the outcome
    groupby_df = df.groupby('date').agg({'outcome': 'sum'}).reset_index()

    plt.plot(groupby_df['date'], groupby_df['outcome'], label='Prevalence', color='blue')


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
    return df, undetected, total_runs, regular_ttd, static_ttd, spc_ttd3, spc_ttd5, spc_ttd7