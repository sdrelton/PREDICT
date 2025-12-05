from PREDICT import PREDICT
from PREDICT.Metrics import Accuracy, CalibrationSlope, CoxSnellR2, CITL, AUROC, AUPRC, F1Score, Precision, Recall, Sensitivity, Specificity, OE
from sklearn.linear_model import LogisticRegression

import pandas as pd
import datetime as dt
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import expit
import itertools
import os
from sklearn.calibration import calibration_curve
import matplotlib.cm as cm
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import binom
import matplotlib.patches as mpatches


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
            filename = f"../docs/images/monitoring/bayesian_coefs/bayesian_coefs_{model_name + '_' if model_name is not None else ''}chunk{i+1}_plot.png"
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

            filename = f"../docs/images/monitoring/bayesian_coefs/bayesian_coefs_{model_name + '_' if model_name else ''}chunk{i+1}_plot.png"
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
    Plot the number of people per month who have over x% risk of the outcome.
    Args:
        threshold (float) : Risk threshold value. Defaults to 0.1.
        model_type (str) : String of model name e.g. 'qrisk'. Defaults to 'qrisk2'.
        gender (str) : If using the qrisk model pick between the male and female model e.g. "female". Defaults to ''.

    """
    def count_over_threshold(df, date_col, threshold=0.1):
        df[date_col] = pd.to_datetime(df[date_col])
        # filter rows
        filtered = df[df['prediction'] > threshold] # new_prediction
        # group by month
        monthly = filtered.groupby(pd.Grouper(key=date_col, freq='M')).size()
        return monthly
    
    plt.figure(figsize=(10,6))

    if gender != '':
        gender = gender+'_'

    if os.path.exists(f"probs_and_outcomes/Baseline_{model_type}_{gender}predictions_and_outcomes.csv"):
        baseline = pd.read_csv(f"probs_and_outcomes/Baseline_{model_type}_{gender}predictions_and_outcomes.csv")
        counts1 = count_over_threshold(baseline, 'date', threshold=threshold)
        counts1.plot(label='Baseline')
    else:
        print(f"File 'probs_and_outcomes/Baseline_{model_type}_{gender}predictions_and_outcomes.csv' for baseline predictions and outcomes does not exist.")

    if os.path.exists(f"probs_and_outcomes/Regular Testing_{model_type}_{gender}predictions_and_outcomes.csv"):
        regular = pd.read_csv(f"probs_and_outcomes/Regular Testing_{model_type}_{gender}predictions_and_outcomes.csv")
        counts2 = count_over_threshold(regular, 'date', threshold=threshold)
        counts2.plot(label='Regular Testing', alpha=0.6, linestyle='--')
    else:
        print("File for regular testing predictions and outcomes does not exist.")

    if os.path.exists(f"probs_and_outcomes/Static Threshold_{model_type}_{gender}predictions_and_outcomes.csv"):
        static = pd.read_csv(f"probs_and_outcomes/Static Threshold_{model_type}_{gender}predictions_and_outcomes.csv")
        counts3 = count_over_threshold(static, 'date', threshold=threshold)
        counts3.plot(label='Static Threshold', alpha=0.6, linestyle=':')
    else:
        print("File for static threshold predictions and outcomes does not exist.")

    if os.path.exists(f"probs_and_outcomes/SPC_{model_type}_{gender}predictions_and_outcomes.csv"):
        spc = pd.read_csv(f"probs_and_outcomes/SPC_{model_type}_{gender}predictions_and_outcomes.csv")
        counts4 = count_over_threshold(spc, 'date', threshold=threshold)
        counts4.plot(label='SPC', alpha=0.6, linestyle='-.')
    else:
        print("File for SPC predictions and outcomes does not exist.")

    if os.path.exists(f"probs_and_outcomes/Bayesian_{model_type}_{gender}predictions_and_outcomes.csv"):
        bayesian = pd.read_csv(f"probs_and_outcomes/Bayesian_{model_type}_{gender}predictions_and_outcomes.csv")
        counts5 = count_over_threshold(bayesian, 'date', threshold=threshold)
        counts5.plot(label='Bayesian', alpha=0.6)
    else:
        print("File for bayesian predictions and outcomes does not exist.")
    if os.path.exists(f"probs_and_outcomes/Baseline_{model_type}_{gender}predictions_and_outcomes.csv"):
        # plot the true number of people who have a heart attack or stroke
        baseline_pos_out = baseline[baseline['outcome'] == 1]
        monthly_true = baseline_pos_out.groupby(pd.Grouper(key='date', freq='M')).size()
        monthly_true.plot(label='True Positive Outcome', alpha=0.3)

    plt.xlabel('Date')
    plt.ylim(-50, None)
    plt.ylabel(f'Number of people with >{int(threshold*100)}% probability')
    plt.title(f'Count of People with Over {int(threshold*100)}% Risk of Outcome')
    plt.legend()
    plt.savefig(f"../docs/images/performance_comparison/{model_type}_{gender}count_over_{int(threshold*100)}%_risk.png", dpi=600, bbox_inches='tight')
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
    if gender != '':
        gender = gender+'_'
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
        fig.savefig(f"../docs/images/performance_comparison/{model_type}_{gender}{metric_choice}.png", dpi=600, bbox_inches='tight')
        plt.show()


def plot_calibration_yearly(model, method_list = ['Baseline', 'Regular Testing', 'Static Threshold', 'SPC', 'Bayesian'], gender=''):
    """Plots the calibration slope each year as one plot with each line as a different PREDICT method.

    Args:
        model (str): Name of the model used e.g. 'qrisk'.
        method_list (list): List of the methods to plot the yearly calibration slope of.
        gender (str): If using a model separated by gender include a string e.g. 'female'. Defaults to ''.
    """
    if gender != '':
        gender = gender+'_'
    for method in method_list:
        if os.path.exists(f"probs_and_outcomes/{method}_{model}_{gender}predictions_and_outcomes.csv"):
            continue
        else:
            print(f"'probs_and_outcomes/{method}_{model}_{gender}predictions_and_outcomes.csv' file does not exist... \nYearly calibration plotting cancelled.\nmethod_list should only contain methods which you have the ...predictions_and_outcomes.csv file for.")
            return
        

        
    method = 'Baseline'
    df = pd.read_csv(f'probs_and_outcomes/{method}_{model}_{gender}predictions_and_outcomes.csv')

    # convert the date column to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Plot a calibration slope for each year:

    df['year'] = df['date'].dt.year

    def bin_stats(preds, y, n_bins=10):
        # decile bins (quantile); duplicates='drop' handles ties
        bins = pd.qcut(preds, q=n_bins, duplicates='drop')
        g = pd.DataFrame({'pred': preds, 'y': y, 'bin': bins}).groupby('bin')
        mean_pred = g['pred'].mean()
        obs = g['y'].mean()
        count = g['y'].count()
        se = np.sqrt(obs * (1 - obs) / count)
        ci_low = (obs - 1.96 * se).clip(0, 1)
        ci_high = (obs + 1.96 * se).clip(0, 1)
        return pd.DataFrame({
            'mean_pred': mean_pred,
            'obs': obs,
            'n': count,
            'ci_low': ci_low,
            'ci_high': ci_high
        }).dropna().reset_index(drop=True)

    def compute_cslope(preds, y):
        eps = 1e-6
        p = np.clip(preds, eps, 1 - eps)
        logit_p = np.log(p / (1 - p)).reshape(-1, 1)
        try:
            lr = LogisticRegression(solver='lbfgs', fit_intercept=True, max_iter=200)
            lr.fit(logit_p, y)
            return float(lr.coef_[0][0])
        except Exception:
            return np.nan

    # Settings
    n_bins = 10
    lowess_frac = None  # set to None to disable LOWESS

    years = sorted(df['year'].unique())
    palette = sns.color_palette('tab10', 5)

    for year in years:
        fig, ax = plt.subplots(figsize=(9, 6))
        legend_handles = []
        legend_labels = []
        plotted_any = False
        for i, method in enumerate(method_list):
            df = pd.read_csv(f'probs_and_outcomes/{method}_{model}_{gender}predictions_and_outcomes.csv')
            df['date'] = pd.to_datetime(df['date'])
            df['year'] = df['date'].dt.year
            year_df = df[df['year'] == year]

            if year_df.empty:
                continue

            preds = year_df['prediction'].values
            y_true = year_df['outcome'].values

            stats = bin_stats(preds, y_true, n_bins=n_bins)
            if stats.empty:
                continue

            color = palette[i]

            # Connect binned points with a line
            ax.plot(stats['mean_pred'], stats['obs'],
                    linewidth=1.5, linestyle='-', zorder=3)

            # # Draw the points
            # ax.plot(stats['mean_pred'], stats['obs'],
            #         color=color, marker=None, markersize=4, linestyle='None', zorder=4)

            # 4) Optional LOWESS smooth (continuous curve from full data)
            if lowess_frac is not None and len(preds) >= 10:
                smooth = lowess(y_true, preds, frac=lowess_frac, return_sorted=True)
                ax.plot(smooth[:, 0], smooth[:, 1], color=color, linewidth=1.0, alpha=0.8, linestyle='--', zorder=1)

            # 5) Compute calibration slope and add legend entry
            cslope = compute_cslope(preds, y_true)
            label = f"{method} (slope={cslope:.2f})" if np.isfinite(cslope) else f"{method} (slope=NA)"
            ln, = ax.plot([], [], color=color, linewidth=1.5)  # invisible handle for legend
            legend_handles.append(ln)
            legend_labels.append(label)

            plotted_any = True

            if not plotted_any:
                plt.close(fig)
                continue

        # perfect calibration line
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1, label='Perfect')

        ax.set_title(f'Calibration in {year}')
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Observed fraction of positives')
        ax.set_xlim(0, 0.3)
        ax.set_ylim(0, 0.3)
        ax.grid(True, linestyle=':', alpha=0.6)

        ax.legend(legend_handles, legend_labels, title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

        plt.savefig(f"../docs/images/calibration_slopes/{model}_{gender}cslope_{year}.png", dpi=600, bbox_inches='tight')

def plot_predictor_distributions(df, predictors, plot_type, model_name):
    """Plots the distributions of the predictors, can choose from using a violin plot, a stacked bar chart
    or a percentage stacked barchart.
    One bar is plotted for each month.

    Args:
        df (pd.DataFrame): Dataframe where predictors are columns and rows are individual visits.
        predictors (list): List of the predictors to plot.
        plot_type (str): What type of plot to draw, either 'violin', 'stacked_bar' or 'stacked_perc'.
        model_name (str): Name of the model e.g. 'qrisk2_female', used to name the saved plots.
    """
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year

    if plot_type == 'violin':
        for predictor in predictors:
            print(f"Plotting distribution per year for {predictor}")  
            df['month_year_label'] = df['date'].dt.strftime('%b %Y')          # 'Jan 2020'

            # make it a categorical with chronological order
            order = sorted(df['month_year_label'].unique(), key=lambda x: pd.to_datetime(x, format='%b %Y'))
            df['month_year_label'] = pd.Categorical(df['month_year_label'], categories=order, ordered=True)

            plt.violinplot(x='month_year_label', y=f'{predictor}', data=df)#, inner='box', cut=0)
            plt.title(f'Distribution of {predictor} by month and year')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.show()
            plt.savefig(f"../docs/images/predictor_distributions/{predictor}_{model_name}_violinplot.png", dpi=600, bbox_inches='tight')
        
    if plot_type == 'stacked_bar':
        for bin_pred in predictors:
            print(f"Printing stacked bar charts for {bin_pred}")
            
            # Create month-year label
            df['month_year_label'] = df['date'].dt.strftime('%b %Y')
            
            # Ensure chronological order
            order = sorted(df['month_year_label'].unique(), 
                        key=lambda x: pd.to_datetime(x, format='%b %Y'))
            df['month_year_label'] = pd.Categorical(df['month_year_label'], categories=order, ordered=True)
            
            # Aggregate counts of 0s and 1s per month
            counts = df.groupby(['month_year_label', bin_pred]).size().unstack(fill_value=0)
            
            # Plot stacked bar chart
            counts.plot(kind='bar', stacked=True, 
                        color=['darkblue', 'lightblue'], figsize=(10,6))
            
            # Add legend
            top_bar = mpatches.Patch(color='darkblue', label=f'{bin_pred} = 0')
            bottom_bar = mpatches.Patch(color='lightblue', label=f'{bin_pred} = 1')
            plt.legend(handles=[top_bar, bottom_bar])
            
            plt.title(f"Distribution of {bin_pred} over time")
            plt.xlabel("Month-Year")
            plt.ylabel("Count")
            
            # Save before showing
            plt.savefig(f"../docs/images/predictor_distributions/{bin_pred}_{model_name}_stacked_barplot.png", 
                        dpi=600, bbox_inches='tight')
            plt.show()
            plt.close()

    if plot_type == 'stacked_perc':
        for bin_pred in predictors:
            print(f"Printing stacked percentage bar charts for {bin_pred}")
            
            # Create month-year label
            df['month_year_label'] = df['date'].dt.strftime('%b %Y')
            
            # Ensure chronological order
            order = sorted(df['month_year_label'].unique(), 
                        key=lambda x: pd.to_datetime(x, format='%b %Y'))
            df['month_year_label'] = pd.Categorical(df['month_year_label'], categories=order, ordered=True)
            
            # Aggregate counts of 0s and 1s per month
            counts = df.groupby(['month_year_label', bin_pred]).size().unstack(fill_value=0)
            
            # Convert to percentages
            percentages = counts.div(counts.sum(axis=1), axis=0) * 100
            
            # Plot stacked percentage bar chart
            percentages.plot(kind='bar', stacked=True, 
                            color=['darkblue', 'lightblue'], figsize=(10,6))
            
            # Add legend
            top_bar = mpatches.Patch(color='darkblue', label=f'{bin_pred} = 0')
            bottom_bar = mpatches.Patch(color='lightblue', label=f'{bin_pred} = 1')
            plt.legend(handles=[top_bar, bottom_bar])
            
            plt.title(f"Percentage distribution of {bin_pred} over time")
            plt.xlabel("Month-Year")
            plt.ylabel("Percentage")
            
            # Save before showing
            plt.savefig(f"../docs/images/predictor_distributions/{bin_pred}_{model_name}_stacked_percentage_barplot.png", 
                        dpi=600, bbox_inches='tight')
            plt.show()
            plt.close()