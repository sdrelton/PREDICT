from PREDICT import PREDICT
from PREDICT.Metrics import Accuracy, CalibrationSlope, CoxSnellR2, CITL, AUROC, AUPRC, F1Score, Precision, Recall, Sensitivity, Specificity, OE

import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
#plt.use("Agg")
from scipy.special import expit
import itertools
import seaborn as sns
import os


def AccuracyPlot(log, recalthreshold=None):
    """Plot the accuracy of the model over time.

    Args:
        log (dict): Log of model metrics over time and when the model was updated.
        recalthreshold (float, int, optional): Threshold to trigger recalibration.
    """
    plt.figure()
    timesteps = list(log['Accuracy'].keys())
    accuracy_values = list(log['Accuracy'].values())

    plt.plot(timesteps, [acc * 100 for acc in accuracy_values], label='Accuracy', marker='o')

    # Add dashed line to indicate when the model was recalibrated
    if 'Model Updated' in log:
        plt.vlines(list(log['Model Updated'].keys()), ymin=min(accuracy_values) * 100, ymax=max(accuracy_values) * 100, 
                colors='r', linestyles='dashed', label='Model Updated')

    # Add recalibration threshold details
    if recalthreshold is not None:
        plt.text(timesteps[0], recalthreshold * 100 + 1, f'Recalibration Threshold: {recalthreshold * 100}%', fontsize=10, color='grey')
        plt.hlines(recalthreshold * 100, min(timesteps), max(timesteps), colors='grey', linestyles='dashed', label='Recalibration Threshold')

    plt.legend(loc='lower right', fontsize=8, markerscale=0.8, frameon=True)

    plt.xlabel('Timesteps')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=90)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


def CalibrationSlopePlot(log):
    """Plot the calibration slope of the model over time.

    Args:
        log (dict): Log of model metrics over time and when the model was updated.
    """
    plt.figure()
    plt.plot(log['CalibrationSlope'].keys(), log['CalibrationSlope'].values(), label='Calibration Slope')

    plt.axhline(y=1, color='black', linestyle='--', label='Ideal Calibration Slope')
    if max(log['CalibrationSlope'].values()) > 1:
        plt.annotate('', xy=(max(log['CalibrationSlope'].keys()), max(log['CalibrationSlope'].values())), xytext=(max(log['CalibrationSlope'].keys()), 1.0001),
                    arrowprops=dict(facecolor='green', shrink=0.05))
        plt.text(min(log['CalibrationSlope'].keys()), ((max(log['CalibrationSlope'].values())+1)/2), 'Overestimation', fontsize=10, color='green')
    if min(log['CalibrationSlope'].values()) < 1:
        plt.annotate('', xy=(max(log['CalibrationSlope'].keys()), min(log['CalibrationSlope'].values())), xytext=(max(log['CalibrationSlope'].keys()), 0.9999),
                    arrowprops=dict(facecolor='red', shrink=0.05))
        plt.text(min(log['CalibrationSlope'].keys()), ((min(log['CalibrationSlope'].values())+1)/2), 'Underestimation', fontsize=10, color='red')

    # Add dashed line to indicate when the model was recalibrated
    if 'Model Updated' in log:
        plt.vlines(log['Model Updated'].keys(), min(log['CalibrationSlope'].values()), max(log['CalibrationSlope'].values()), colors='r', linestyles='dashed', label='Model Updated')
    plt.xlabel('Timesteps')
    plt.ylabel('Calibration Slope')
    
    plt.legend(loc='upper right', fontsize=8, markerscale=0.8, frameon=True)
    plt.xticks(rotation=90)
    plt.show()

def CoxSnellPlot(log):
    """Plot the Cox-Snell R^2 of the model over time.

    Args:
        log (dict): Log of model metrics over time and when the model was updated.
    """
    plt.figure()
    plt.plot(log['CoxSnellR2'].keys(), log['CoxSnellR2'].values(), label='Cox and Snell R2')
    if 'Model Updated' in log:
        plt.vlines(log['Model Updated'].keys(), min(log['CoxSnellR2'].values())-0.2, max(log['CoxSnellR2'].values())+0.2, colors='r', linestyles='dashed', label='Model Updated')
    plt.ylim(min(log['CoxSnellR2'].values())-0.01, max(log['CoxSnellR2'].values())+0.01)
    plt.xlabel('Timesteps')
    plt.ylabel('Cox and Snell R2')
    plt.legend(loc='lower left', fontsize=8, markerscale=0.8, frameon=True)
    plt.xticks(rotation=90)
    plt.show()

def CITLPlot(log):
    """Plot the Calibration-in-the-Large of the model over time.

    Args:
        log (dict): Log of model metrics over time and when the model was updated.
    """
    plt.figure()
    plt.plot(log['CITL'].keys(), log['CITL'].values(), label='CITL')
    plt.axhline(y=0, color='black', linestyle='--', label='Ideal CITL')
    if max(log['CITL'].values()) > 0:
        plt.annotate('', xy=(max(log['CITL'].keys()), 0.35), xytext=(max(log['CITL'].keys()), 0.1),
                    arrowprops=dict(facecolor='green', shrink=0.05))
        plt.text(min(log['CITL'].keys()), 0.3, 'Underestimation', fontsize=10, color='green')
    if min(log['CITL'].values()) < 0:
        plt.annotate('', xy=(max(log['CITL'].keys()), -0.3), xytext=(max(log['CITL'].keys()), -0.1),
                    arrowprops=dict(facecolor='red', shrink=0.05))
        plt.text(min(log['CITL'].keys()), -0.35, 'Overestimation', fontsize=10, color='red')
    if 'Model Updated' in log:
        plt.vlines(log['Model Updated'].keys(), min(log['CITL'].values())-0.2, max(log['CITL'].values())+0.2, colors='r', linestyles='dashed', label='Model Updated')
    plt.ylim(min(log['CITL'].values()), max(log['CITL'].values()))
    plt.title('CITL')
    plt.xlabel('Timesteps')
    plt.ylabel('CITL')
    plt.legend(loc='lower left', fontsize=8, markerscale=0.8, frameon=True)
    plt.xticks(rotation=90)
    plt.show()

def AUROCPlot(log):
    """Plot the AUROC of the model over time.

    Args:
        log (dict): Log of model metrics over time and when the model was updated.
    """
    plt.figure()
    plt.plot(log['AUROC'].keys(), log['AUROC'].values(), label='AUROC')
    if 'Model Updated' in log:
        plt.vlines(log['Model Updated'].keys(), min(log['AUROC'].values())-0.2, max(log['AUROC'].values())+0.2, colors='r', linestyles='dashed', label='Model Updated')
    plt.xlabel('Timesteps')
    plt.ylabel('AUROC')
    plt.legend(loc='lower left', fontsize=8, markerscale=0.8, frameon=True)
    plt.xticks(rotation=90)
    plt.show()

def AUPRCPlot(log):
    """Plot the AUPRC of the model over time.

    Args:
        log (dict): Log of model metrics over time and when the model was updated.
    """
    plt.figure()
    plt.plot(log['AUPRC'].keys(), log['AUPRC'].values(), label='AUPRC')
    if 'Model Updated' in log:
        plt.vlines(log['Model Updated'].keys(), min(log['AUPRC'].values())-0.2, max(log['AUPRC'].values())+0.2, colors='r', linestyles='dashed', label='Model Updated')
    plt.xlabel('Timesteps')
    plt.ylabel('AUPRC')
    plt.legend(loc='lower left', fontsize=8, markerscale=0.8, frameon=True)
    plt.xticks(rotation=90)
    plt.show()

def F1ScorePlot(log):
    """Plot the F1 Score of the model over time.

    Args:
        log (dict): Log of model metrics over time and when the model was updated.
    """
    plt.figure()
    plt.plot(log['F1score'].keys(), log['F1score'].values(), label='F1 Score')
    if 'Model Updated' in log:
        plt.vlines(log['Model Updated'].keys(), min(log['F1score'].values())-0.2, max(log['F1score'].values())+0.2, colors='r', linestyles='dashed', label='Model Updated')
    plt.xlabel('Timesteps')
    plt.ylabel('F1 Score')
    plt.legend(loc='lower left', fontsize=8, markerscale=0.8, frameon=True)
    plt.xticks(rotation=90)
    plt.show()

def PrecisionPlot(log):
    """Plot the Precision of the model over time.

    Args:
        log (dict): Log of model metrics over time and when the model was updated.
    """
    plt.figure()
    plt.plot(log['Precision'].keys(), log['Precision'].values(), label='Precision')
    if 'Model Updated' in log:
        plt.vlines(log['Model Updated'].keys(), min(log['Precision'].values())-0.2, max(log['Precision'].values())+0.2, colors='r', linestyles='dashed', label='Model Updated')
    plt.xlabel('Timesteps')
    plt.ylabel('Precision')
    plt.legend(loc='lower left', fontsize=8, markerscale=0.8, frameon=True)
    plt.xticks(rotation=90)
    plt.show()

def SensitivityPlot(log):
    """Plot the Sensitivity of the model over time.

    Args:
        log (dict): Log of model metrics over time and when the model was updated.
    """
    plt.figure()
    plt.plot(log['Sensitivity'].keys(), log['Sensitivity'].values(), label='Sensitivity')
    if 'Model Updated' in log:
        plt.vlines(log['Model Updated'].keys(), min(log['Sensitivity'].values())-0.2, max(log['Sensitivity'].values())+0.2, colors='r', linestyles='dashed', label='Model Updated')
    plt.xlabel('Timesteps')
    plt.ylabel('Sensitivity')
    plt.legend(loc='lower left', fontsize=8, markerscale=0.8, frameon=True)
    plt.xticks(rotation=90)
    plt.show()

def SpecificityPlot(log):
    """Plot the Specificity of the model over time.

    Args:
        log (dict): Log of model metrics over time and when the model was updated.
    """
    plt.figure()
    plt.plot(log['Specificity'].keys(), log['Specificity'].values(), label='Specificity')
    if 'Model Updated' in log:
        plt.vlines(log['Model Updated'].keys(), min(log['Specificity'].values())-0.2, max(log['Specificity'].values())+0.2, colors='r', linestyles='dashed', label='Model Updated')
    plt.xlabel('Timesteps')
    plt.ylabel('Specificity')
    plt.legend(loc='lower left', fontsize=8, markerscale=0.8, frameon=True)
    plt.xticks(rotation=90)
    plt.show()

def OEPlot(log):
    """Plot the observation to expectation ratio of the model over time.

    Args:
        log (dict): Log of model metrics over time and when the model was updated.
    """
    plt.figure()
    plt.plot(log['O/E'].keys(), log['O/E'].values(), label='O/E')
    if 'Model Updated' in log:
        plt.vlines(log['Model Updated'].keys(), min(log['O/E'].values())-0.2, max(log['O/E'].values())+0.2, colors='r', linestyles='dashed', label='Model Updated')
    plt.xlabel('Timesteps')
    plt.ylabel('O/E')
    
    plt.legend(loc='lower left', fontsize=8, markerscale=0.8, frameon=True)
    plt.xticks(rotation=90)
    plt.show()


def NormalisedSumOfDiffPlot(log):
    """Plot the error of the model over time.

    Args:
        log (dict): Log of model metrics over time and when the model was updated.
    """
    plt.figure()
    plt.plot(log['NormSumOfDifferences'].keys(), log['NormSumOfDifferences'].values(), label='Normalised Sum Of Differences')
    if 'Model Updated' in log:
        plt.vlines(log['Model Updated'].keys(), min(log['NormSumOfDifferences'].values())-0.2, max(log['NormSumOfDifferences'].values())+0.2, colors='r', linestyles='dashed', label='Model Updated')
    plt.xlabel('Timesteps')
    plt.ylabel('Sum of Differences Error')
    plt.hlines(0, min(log['NormSumOfDifferences'].keys()), max(log['NormSumOfDifferences'].keys()), colors='black', linestyles='dashed', label='No error')
    plt.legend(loc='upper right', fontsize=8, markerscale=0.8, frameon=True)
    plt.xticks(rotation=90)
    plt.show()

def ErrorSPCPlot(log, model):
    """Plots the error over time as a statistical process control chart with upper control 
    limits indicating warning and danger zones when model performance drops.

    Args:
        log (dict): Log of model metrics over time and when the model was updated.
        model (PREDICTModel): The model to evaluate, must have a predict method.
    """
    fig, ax = plt.subplots()  # Use subplots to get an explicit axes object

    error_df = pd.DataFrame(list(log['NormSumOfDifferences'].items()), columns=['Date', 'NormSumOfDifferences'])
    ax.plot(error_df['Date'], error_df['NormSumOfDifferences'], marker='o', label='Data')

    # Set where the recalibration zones end for plot aesthetics
    ucl = error_df['NormSumOfDifferences'].max() + (0.1 * error_df['NormSumOfDifferences'].max())
    lcl = error_df['NormSumOfDifferences'].min() - (0.1 * error_df['NormSumOfDifferences'].min())

    ucl = ax.get_ylim()[1]
    lcl = ax.get_ylim()[0]

    ax.axhline(model.mean_error, color='black', linestyle='--', label='Mean (X-bar)')
    ax.axhline(model.u2sdl, color='black', linestyle='-')
    ax.axhline(model.u3sdl, color='black', linestyle='-')
    ax.axhline(model.l2sdl, color='black', linestyle='-')
    ax.axhline(model.l3sdl, color='black', linestyle='-')

    if 'Model Updated' in log:
        ax.vlines(log['Model Updated'].keys(), lcl, ucl, colors='r', linestyles='dashed', label='Model Updated')

    # Colour the safe and warning zones
    ax.fill_between(error_df['Date'], model.u3sdl, ucl, color='red', alpha=0.2, label='Recalibration zone')
    ax.fill_between(error_df['Date'], model.u2sdl, model.u3sdl, color='yellow', alpha=0.2, label='Warning zone')
    ax.fill_between(error_df['Date'], model.l2sdl, model.u2sdl, color='green', alpha=0.2, label='Safe zone')
    ax.fill_between(error_df['Date'], model.l2sdl, model.l3sdl, color='yellow', alpha=0.2)
    ax.fill_between(error_df['Date'], model.l3sdl, lcl, color='red', alpha=0.2)

    ax.set_title('SPC Chart for Error')
    ax.set_xlabel('Date')
    ax.set_ylabel('Normalised Error')
    plt.xticks(rotation=90)
    plt.legend(fontsize=8, markerscale=0.8, frameon=True)
    plt.grid(False)
    plt.show()


def MonitorChangeSPC(input_data, trackCol, timeframe, windowSize, largerSD=3, smallerSD=2):
    """Generate a statistical process control chart to observe data changes over time.
    Plot shows prevalence or mean of a dataframe column over time with control limits for ± x and y 
    standard deviations from the mean (where x and y default to 2 and 3 respectively).
    This function is useful for tracking changes that might control to model error increasing.

    Args:
        input_data (pd.DataFrame): The input data to monitor data changes.
        trackCol (str): Column of input data to monitor.
        timeframe (str): How often to plot the data points of the tracked variable. Can be 'Day', 'Week', 'Month' or 'Year'.
        windowSize (int): How many timeframes to use as a the rolling control limit window size e.g. if timeframe is 'week' and 
            the window_size = 4 then the window covers 4 weeks.
        largerSD (float): Red line upper and lower most control limits. Defaults to 3.
        smallerSD (float): Yellow line inner control limts. Defaults to 2.
        

    Raises:
        ValueError: If timeframe variable is not 'Day', 'Week', 'Month', or 'Year'.
    """
    plt.figure()

    if smallerSD > largerSD:
        raise ValueError(f"smallerSD must be smaller than largerSD. smallerSD: {smallerSD} > largerSD: {largerSD}")

    # Group data by date and calculate the mean of the variable at each date
    data_prev = input_data.groupby('date').agg({trackCol: 'mean'}).reset_index()
    data_prev.rename(columns={trackCol: 'daily_mean'}, inplace=True) 

    if timeframe == 'Week':
        period = 'W'
    elif timeframe == 'Day':
        period = 'D'
    elif timeframe == 'Month':
        period = 'M'
    elif timeframe == 'Year':
        period = 'Y'
    else:
        raise ValueError("timeframe variable must be 'Day', 'Week', 'Month', or 'Year'")

    # Assign time period (e.g., start of the week) for each date
    data_prev[timeframe] = data_prev['date'].dt.to_period(period).apply(lambda r: r.start_time)

    # Group by the defined timeframe and calculate the mean for each group
    data_time_grouped = data_prev.groupby(timeframe).agg({
        'daily_mean': 'mean' 
    }).reset_index()

    data_time_grouped['rolling_mean'] = None
    data_time_grouped['ucl'] = None
    data_time_grouped['lcl'] = None
    data_time_grouped['ucl2sd'] = None
    data_time_grouped['lcl2sd'] = None

    for i in range(len(data_time_grouped)):
        # Use a rolling window for control limit calculations
        start_index = max(0, i - windowSize + 1)
        subset = data_time_grouped.iloc[start_index:i+1]

        # Calculate rolling mean and standard deviation
        rolling_mean = subset['daily_mean'].mean()
        rolling_std_dev = subset['daily_mean'].std(ddof=0)

        # Update control limits
        data_time_grouped.loc[i, 'rolling_mean'] = rolling_mean
        data_time_grouped.loc[i, 'ucl'] = rolling_mean + largerSD * rolling_std_dev
        data_time_grouped.loc[i, 'lcl'] = max(rolling_mean - largerSD * rolling_std_dev, 0)
        data_time_grouped.loc[i, 'ucl2sd'] = rolling_mean + smallerSD * rolling_std_dev
        data_time_grouped.loc[i, 'lcl2sd'] = max(rolling_mean - smallerSD * rolling_std_dev, 0)

    plt.plot(data_time_grouped[timeframe], data_time_grouped['rolling_mean'], color='black', linestyle='-', alpha=0.6)
    plt.plot(data_time_grouped[timeframe], data_time_grouped['ucl'], color='red', linestyle='-', alpha=0.6, label=f'±{largerSD}σ from the mean')
    plt.plot(data_time_grouped[timeframe], data_time_grouped['lcl'], color='red', linestyle='-', alpha=0.6)
    plt.plot(data_time_grouped[timeframe], data_time_grouped['ucl2sd'], color='orange', linestyle='-', alpha=0.6, label=f'±{smallerSD}σ from the mean')
    plt.plot(data_time_grouped[timeframe], data_time_grouped['lcl2sd'], color='orange', linestyle='-', alpha=0.6)
    plt.plot(data_time_grouped[timeframe], data_time_grouped['daily_mean'], marker='o', label=f'{timeframe} Average')

    plt.title(f'Moving Control Chart for {trackCol.capitalize()} Over Time')
    plt.xlabel('Date')
    if input_data[trackCol].isin([0, 1]).all():
        plt.ylabel(f'Prevalence of {trackCol.capitalize()}')
    else:
        plt.ylabel(f'Mean of {trackCol.capitalize()}')
    plt.xticks(rotation=90)
    plt.legend(fontsize=8, markerscale=0.8, frameon=True)
    plt.grid(True)
    plt.show()

def PredictorBasedPlot(log, x_axis_min=None, x_axis_max=None, predictor=None, outcome="outcome", show_legend=True):
    """Plots the probability of an outcome given a specific predictor.
    Note: this is only suitable for the BayesianModel and .addLogHook(TrackBayesianCoefs(model)) must be used.

    Args:
        log (dict): Log of model metrics over time and when the model was updated.
        x_axis_min (float, optional): Minimum value for the x axis representing the predictor. Defaults to None.
        x_axis_max (float, optional): Maximum value for the x axis representing the predictor. Defaults to None.
        predictor (str, optional): Name of the predictor to assess. Defaults to None.
        outcome (str, optional): Name of the outcome being predicted. Defaults to "outcome".
        show_legend (bool, optional): Whether to show the legend. Defaults to True.

    Raises:
        ValueError: Raises error if x_axis_min is not provided.
        ValueError: Raises error if x_axis_max is not provided.
        ValueError: Raises error if predictor is not provided.
    """
    if x_axis_min is None:
        raise ValueError("x_axis_min is None, minimum x axis value required for the plot.")
    if x_axis_max is None:
        raise ValueError("x_axis_max is None, maximum x axis value required for the plot.")
    if predictor is None:
        raise ValueError("predictor is None, a predictor is required to determine probability of the outcome.")

    plt.figure()
    bayesianCoefs = log["BayesianCoefficients"]
    timestamps = list(bayesianCoefs.keys())
    x_axis_values = list(range(x_axis_min, x_axis_max+1))
    for timestamp in timestamps:
        specific_coefs = bayesianCoefs[pd.Timestamp(timestamp)]
        if specific_coefs[predictor] is not None:
            mean_coef = specific_coefs[predictor][0]
            intercept = specific_coefs["Intercept"][0]
            probs = []
            for value in x_axis_values:
                linear_function = intercept + mean_coef * value
                prob = expit(linear_function)
                probs.append(prob)
            plt.plot(x_axis_values, probs, label=timestamp, alpha=0.5)

    plt.xlabel(f"{predictor}")
    plt.ylabel(f"Probability of {outcome}")
    legend = plt.legend(title="Time", fontsize=8, markerscale=0.8, frameon=True)
    legend.set_visible(show_legend)
    plt.grid(True)
    plt.show()



def BayesianCoefsPlot(log, model_name=None, max_predictors_per_plot=10):
    """
    Plots the mean coefficients (with standard deviation as the error bar) of the Bayesian model over time.
    Note: this is only suitable for the BayesianModel and .addLogHook(TrackBayesianCoefs(model)) must be used.

    Args:
        log (dict, pd.DataFrame): Log or dataframe of model metrics over time and when the model was updated.
        model_name (str, optional): Name of model or domain used in filename e.g. 'COVID_data_simulation'.
        max_predictors_per_plot (int): Max number of predictors per plot to avoid clutter.
    """
    if isinstance(log, dict):
        bayesianCoefs = log["BayesianCoefficients"]
        timestamps = list(bayesianCoefs.keys())

        # Collect all predictors
        all_predictors = sorted({key for timestamp in timestamps for key in bayesianCoefs[pd.Timestamp(timestamp)].keys()})

        # Prepare DataFrame
        data = []
        for timestamp in timestamps:
            specific_coefs = bayesianCoefs[pd.Timestamp(timestamp)]
            for predictor, (mean_coef, std_coef) in specific_coefs.items():
                data.append({
                    "Timestamp": pd.Timestamp(timestamp),
                    "Predictor": predictor,
                    "Mean Coef": mean_coef,
                    "Std Coef": std_coef
                })
        coefs_df = pd.DataFrame(data)

        # Split predictors into chunks
        predictor_chunks = [all_predictors[i:i + max_predictors_per_plot] for i in range(0, len(all_predictors), max_predictors_per_plot)]

        # Plot each chunk
        for i, chunk in enumerate(predictor_chunks):
            plt.figure(figsize=(10, 5))
            for predictor in chunk:
                predictor_data = coefs_df[coefs_df['Predictor'] == predictor]
                plt.errorbar(
                    predictor_data['Timestamp'],
                    predictor_data['Mean Coef'],
                    yerr=predictor_data['Std Coef'],
                    fmt='-o',
                    alpha=0.6,
                    label=predictor
                )

            plt.xlabel("Time")
            plt.ylabel("Coefficient")
            plt.title(f"Bayesian Priors Over Time (Predictor Set {i+1})")
            plt.yscale('symlog', linthresh=1)
            plt.xticks(timestamps, rotation=90)
            plt.grid(True)
            plt.legend(title="Predictor", fontsize=8, markerscale=0.8, frameon=True)
            plt.tight_layout()
            filename = f"../docs/images/monitoring/bayesian_coefs/bayesian_coefs_{'_'+model_name if model_name is not None else ''}_chunk{i+1}_plot.png" if model_name else f"bayesian_coefs_chunk{i+1}_plot.png"
            plt.savefig(filename, dpi=600)
            plt.show()

    
    elif isinstance(log, pd.DataFrame):
        df = log.copy()
        df['date'] = pd.to_datetime(df['date'])


        # Parse "(mean, std)" strings into numeric columns
        parsed = []
        for col in df.columns:
            if col != 'date':
                temp = df[col].str.strip("()").str.split(",", expand=True)
                temp.columns = ['Mean Coef', 'Std Coef']
                temp['Mean Coef'] = temp['Mean Coef'].astype(float)
                temp['Std Coef'] = temp['Std Coef'].astype(float)
                temp['Predictor'] = col
                temp['date'] = df['date']
                parsed.append(temp)

        coefs_df = pd.concat(parsed, ignore_index=True)
        date_col = "date"

    

        # Split predictors into chunks
        all_predictors = sorted(coefs_df['Predictor'].unique())
        predictor_chunks = [all_predictors[i:i + max_predictors_per_plot] for i in range(0, len(all_predictors), max_predictors_per_plot)]

        # Plot each chunk
        for i, chunk in enumerate(predictor_chunks):
            plt.figure(figsize=(10, 5))
            for predictor in chunk:
                predictor_data = coefs_df[coefs_df['Predictor'] == predictor]
                plt.errorbar(
                    predictor_data[date_col],
                    predictor_data['Mean Coef'],
                    yerr=predictor_data['Std Coef'],
                    fmt='-o',
                    alpha=0.6,
                    label=predictor
                )

            plt.xlabel("Time")
            plt.ylabel("Coefficient")
            plt.title(f"Bayesian Priors Over Time (Predictor Set {i+1})")
            plt.yscale('symlog', linthresh=1)
            plt.xticks(rotation=90)
            plt.grid(True)
            plt.legend(title="Predictor", fontsize=8, markerscale=0.8, frameon=True)
            plt.tight_layout()

            filename = f"bayesian_coefs_{model_name + '_' if model_name else ''}chunk{i+1}_plot.png"
            plt.savefig(filename, dpi=600)
            plt.show()


    else:
        raise ValueError("Bayesian coefficients need to be either the log or the pd.DataFrame of the coefficients.")
    

def plot_patients_per_month(df, model_type:str, gender:str=''):
    """
    Plots the number of people per month.
    Args:
        df (pd.DataFrame) : DataFrame of patient data.
        model_type (str) : String of model name e.g. 'qrisk'.
        gender (str) : If using the qrisk model pick between the male and female model e.g. "female". Defaults to ''.

    """
    df['date'] = pd.to_datetime(df['date'])

    # Create a 'month' column for grouping
    df['month'] = df['date'].dt.to_period('M')

    # Count number of patients per month
    monthly_patient_count = df.groupby('month').size()

    # Convert PeriodIndex to datetime for plotting
    monthly_patient_count.index = monthly_patient_count.index.to_timestamp()

    plt.figure(figsize=(10, 6))
    monthly_patient_count.plot(kind='bar')
    plt.title("Number of Patients per Month")
    plt.xlabel("Month")
    plt.ylabel("Patient Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"../docs/images/performance_comparison/{model_type}_{gender}_num_patients_per_month.png", dpi=600, bbox_inches='tight')
    plt.show()
        

def plot_count_of_patients_over_threshold_risk(threshold=0.1, model_type='qrisk2', gender=''):
    """
    Plot the number of people per week who have over x% risk of the outcome.
    Args:
        threshold (float) : Risk threshold value. Defaults to 0.1.
        model_type (str) : String of model name e.g. 'qrisk'. Defaults to 'qrisk2'.
        gender (str) : If using the qrisk model pick between the male and female model e.g. "female". Defaults to ''.

    """
    def count_over_threshold(df, date_col, prob_col, threshold=0.1):
        df[date_col] = pd.to_datetime(df[date_col])
        # filter rows
        filtered = df[df[prob_col] > threshold]
        # group by week
        weekly = filtered.groupby(pd.Grouper(key=date_col, freq='W')).size()
        return weekly
    
    plt.figure(figsize=(10,6))

    if os.path.exists(f"Baseline_{model_type}_{gender}_predictions_and_outcomes.csv"):
        baseline = pd.read_csv(f"Baseline_{model_type}_{gender}_predictions_and_outcomes.csv")
        counts1 = count_over_threshold(baseline, 'date', 'Baseline preds', threshold=threshold)
        counts1.plot(label='Baseline')
    else:
        print("File for baseline predictions and outcomes does not exist.")

    if os.path.exists(f"Regular Testing_{model_type}_{gender}_predictions_and_outcomes.csv"):
        regular = pd.read_csv(f"Regular Testing_{model_type}_{gender}_predictions_and_outcomes.csv")
        counts2 = count_over_threshold(regular, 'date', 'Regular Testing preds', threshold=threshold)
        counts2.plot(label='Regular Testing')
    else:
        print("File for regular testing predictions and outcomes does not exist.")

    if os.path.exists(f"Static Threshold_{model_type}_{gender}_predictions_and_outcomes.csv"):
        static = pd.read_csv(f"Static Threshold_{model_type}_{gender}_predictions_and_outcomes.csv")
        counts3 = count_over_threshold(static, 'date', 'Static Threshold preds', threshold=threshold)
        counts3.plot(label='Static Threshold')
    else:
        print("File for static threshold predictions and outcomes does not exist.")

    if os.path.exists(f"SPC_{model_type}_{gender}_predictions_and_outcomes.csv"):
        spc = pd.read_csv(f"SPC_{model_type}_{gender}_predictions_and_outcomes.csv")
        counts4 = count_over_threshold(spc, 'date', 'SPC preds', threshold=threshold)
        counts4.plot(label='SPC')
    else:
        print("File for SPC predictions and outcomes does not exist.")

    if os.path.exists(f"Bayesian_{model_type}_{gender}_predictions_and_outcomes.csv"):
        bayesian = pd.read_csv(f"Bayesian_{model_type}_{gender}_predictions_and_outcomes.csv")
        counts5 = count_over_threshold(bayesian, 'date', 'Bayesian preds', threshold=threshold)
        counts5.plot(label='Bayesian')
    else:
        print("File for bayesian predictions and outcomes does not exist.")

    plt.xlabel('Date')
    plt.ylabel(f'Number of people with >{int(threshold*100)}% probability')
    plt.title(f'Count of People with Over {int(threshold*100)}% Risk of Outcome')
    plt.legend()
    plt.savefig(f"{model_type}_{gender}_count_over_{int(threshold*100)}%_risk.png", dpi=600, bbox_inches='tight')
    plt.show()

def plot_method_comparison_metrics(metrics_df, recalthreshold, model_updates, model_type, gender=''):
    """
    Plot the metric comparison graphs with each line showing a different PREDICT method.
    Args:
        metrics_df (str) : csv file name where performance metrics for each method are saved.
        recalthreshold (float) : static threshold method AUROC threshold.
        model_updates (str) : csv file name where dates of model updates with method names are stored.
        model_type (str) : name of the model used e.g. QRISK2.
        gender (str) : if using the QRISK model, define whether to use male or female. Defaults to ''.

    """
    metrics_df = pd.read_csv(metrics_df)
    model_updates = pd.read_csv(model_updates)
    metrics_df["Time"] = pd.to_datetime(metrics_df["Time"])
    sns.set(font_scale=1.2)
    metric_choices = ["CalibrationSlope", "OE", "CITL"]

    for metric_choice in metric_choices:

        metrics_df["Method"] = metrics_df["Method"].replace({"Static Threshold": f"Static Threshold ({round(recalthreshold, 2)})"})

        fig, ax = plt.subplots(figsize=(14, 7))

        sns.lineplot(
            data=metrics_df,
            x="Time",
            y=metric_choice,
            hue="Method",
            ci=None,
            style="Method",
            ax=ax
        )

        
        model_updates["date"] = pd.to_datetime(model_updates["date"])

        for method in ['Regular Testing', "Static Threshold", "SPC", "Bayesian"]:
            if method in model_updates['method'].values:
                if method == "Regular Testing":
                    marker = 'o'
                    colour = 'orange'
                elif method == "Static Threshold":
                    marker = '|'
                    colour = 'green'
                elif method == "SPC":
                    marker = '^'
                    colour = 'red'
                elif method == "Bayesian":
                    marker = 'D'
                    colour = 'purple'
                subset = model_updates[model_updates["method"] == method]
                ax.scatter(
                    subset["date"],
                    [metrics_df[metric_choice].min()]*len(subset),
                    label=f"{method} update",
                    marker=marker,
                    color=colour,
                    alpha=0.4,
                    s=40
                )

        ax.set_title(metric_choice)
        ax.set_xlabel("Date")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        fig.tight_layout()
        fig.savefig(f"../docs/images/performance_comparison/{model_type}_{gender}_{metric_choice}.png", dpi=600, bbox_inches='tight')
        plt.show()