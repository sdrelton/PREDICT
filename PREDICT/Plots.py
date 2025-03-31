from PREDICT import PREDICT
from PREDICT.Metrics import Accuracy, CalibrationSlope, CoxSnellR2, CITL, AUROC, AUPRC, F1Score, Precision, Recall, Sensitivity, Specificity, OE

import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt


def AccuracyPlot(log, recalthreshold=None):
    """Plot the accuracy of the model over time.

    Args:
        log (dict): Log of model metrics over time and when the model was updated.
        recalthreshold (float, int): Threshold to trigger recalibration. Defaults to None.
    """
    plt.figure()
    plt.plot(log['Accuracy'].keys(), log['Accuracy'].values(), label='Accuracy')

    # Add dashed line to indicate when the model was recalibrated
    if 'Model Updated' in log:
        plt.vlines(log['Model Updated'].keys(), 0, 1, colors='r', linestyles='dashed', label='Model Updated')

    # Add recalibration threshold details
    if recalthreshold is not None:
        plt.text(min(log['Accuracy'].keys()), recalthreshold + 0.01, f'Recalibration Threshold: {recalthreshold * 100}%', fontsize=10, color='grey')
        plt.hlines(recalthreshold, min(log['Accuracy'].keys()), max(log['Accuracy'].keys()), colors='grey', linestyles='dashed', label='Recalibration Threshold')

    plt.legend(loc='lower right')

    plt.xlabel('Timesteps')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=90)
    plt.show()


def CalibrationSlopePlot(log):
    """Plot the calibration slope of the model over time.

    Args:
        log (dict): Log of model metrics over time and when the model was updated.
    """
    plt.figure()
    plt.plot(log['CalibrationSlope'].keys(), log['CalibrationSlope'].values(), label='Calibration Slope')

    plt.axhline(y=1, color='black', linestyle='--', label='Ideal Calibration Slope')
    plt.annotate('', xy=(max(log['CalibrationSlope'].keys()), max(log['CalibrationSlope'].values())), xytext=(max(log['CalibrationSlope'].keys()), 1.0001),
                arrowprops=dict(facecolor='green', shrink=0.05))
    plt.text(min(log['CalibrationSlope'].keys()), ((max(log['CalibrationSlope'].values())+1)/2), 'Overestimation', fontsize=10, color='green')
    plt.annotate('', xy=(max(log['CalibrationSlope'].keys()), min(log['CalibrationSlope'].values())), xytext=(max(log['CalibrationSlope'].keys()), 0.9999),
                arrowprops=dict(facecolor='red', shrink=0.05))
    plt.text(min(log['CalibrationSlope'].keys()), ((min(log['CalibrationSlope'].values())+1)/2), 'Underestimation', fontsize=10, color='red')

    # Add dashed line to indicate when the model was recalibrated
    if 'Model Updated' in log:
        plt.vlines(log['Model Updated'].keys(), min(log['CalibrationSlope'].values()), max(log['CalibrationSlope'].values()), colors='r', linestyles='dashed', label='Model Updated')
    plt.xlabel('Timesteps')
    plt.ylabel('Calibration Slope')
    
    plt.legend(loc='upper right')
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
    plt.legend(loc='lower left')
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
    plt.annotate('', xy=(max(log['CITL'].keys()), 0.35), xytext=(max(log['CITL'].keys()), 0.1),
                arrowprops=dict(facecolor='green', shrink=0.05))
    plt.text(min(log['CITL'].keys()), 0.3, 'Underestimation', fontsize=10, color='green')
    plt.annotate('', xy=(max(log['CITL'].keys()), -0.3), xytext=(max(log['CITL'].keys()), -0.1),
                arrowprops=dict(facecolor='red', shrink=0.05))
    plt.text(min(log['CITL'].keys()), -0.35, 'Overestimation', fontsize=10, color='red')
    if 'Model Updated' in log:
        plt.vlines(log['Model Updated'].keys(), min(log['CITL'].values())-0.2, max(log['CITL'].values())+0.2, colors='r', linestyles='dashed', label='Model Updated')
    plt.ylim(min(log['CITL'].values()), max(log['CITL'].values()))
    plt.title('CITL')
    plt.xlabel('Timesteps')
    plt.ylabel('CITL')
    plt.legend(loc='lower left')
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
    plt.legend(loc='lower left')
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
    plt.legend(loc='lower left')
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
    plt.legend(loc='lower left')
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
    plt.legend(loc='lower left')
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
    plt.legend(loc='lower left')
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
    plt.legend(loc='lower left')
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
    
    plt.legend(loc='lower left')
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
    plt.legend(loc='upper right')
    plt.xticks(rotation=90)
    plt.show()

def ErrorSPCPlot(log, model):
    """Plots the error over time as a statistical process control chart with upper control 
    limits indicating warning and danger zones when model performance drops.

    Args:
        log (dict): Log of model metrics over time and when the model was updated.
        model (PREDICTModel): The model to evaluate, must have a predict method.
    """
    plt.figure()
    error_df = pd.DataFrame(list(log['NormSumOfDifferences'].items()), columns=['Date', 'NormSumOfDifferences'])
    plt.plot(error_df['Date'], error_df['NormSumOfDifferences'], marker='o', label='Data')

    # set where the recalibration zones end for plot aesthetics
    ucl = error_df['NormSumOfDifferences'].max()+(0.1*error_df['NormSumOfDifferences'].max())
    lcl = error_df['NormSumOfDifferences'].min()+(0.1*error_df['NormSumOfDifferences'].min())

    plt.axhline(model.mean_error, color='black', linestyle='--', label='Mean (X-bar)')
    plt.axhline(model.u2sdl, color='black', linestyle='-')
    plt.axhline(model.u3sdl, color='black', linestyle='-')
    plt.axhline(model.l2sdl, color='black', linestyle='-')
    plt.axhline(model.l3sdl, color='black', linestyle='-')

    if 'Model Updated' in log:
        plt.vlines(log['Model Updated'].keys(), lcl, ucl, colors='r', linestyles='dashed', label='Model Updated')

    plt.fill_between(error_df['Date'], model.u3sdl, ucl, color='red', alpha=0.2, label='Recalibration zone')
    plt.fill_between(error_df['Date'], model.u2sdl, model.u3sdl, color='yellow', alpha=0.2, label='Warning zone')
    plt.fill_between(error_df['Date'], model.l2sdl, model.u2sdl, color='green', alpha=0.2, label='Safe zone')
    plt.fill_between(error_df['Date'], model.l2sdl, model.l3sdl, color='yellow', alpha=0.2)
    plt.fill_between(error_df['Date'], model.l3sdl, lcl, color='red', alpha=0.2)



    plt.title('SPC Chart for Error')
    plt.xlabel('Date')
    plt.ylabel('Normalised Error')
    plt.xticks(rotation=90)
    plt.legend()
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
    plt.legend()
    plt.grid(True)
    plt.show()