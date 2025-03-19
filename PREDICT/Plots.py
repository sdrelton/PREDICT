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
    plt.plot(log['O/E'].keys(), log['O/E'].values(), label='O/E')
    if 'Model Updated' in log:
        plt.vlines(log['Model Updated'].keys(), min(log['O/E'].values())-0.2, max(log['O/E'].values())+0.2, colors='r', linestyles='dashed', label='Model Updated')
    plt.xlabel('Timesteps')
    plt.ylabel('O/E')
    
    plt.legend(loc='lower left')
    plt.xticks(rotation=90)
    plt.show()


def LogRegErrorPlot(log):

    plt.plot(log['LogRegError'].keys(), log['LogRegError'].values(), label='LogRegError')
    if 'Model Updated' in log:
        plt.vlines(log['Model Updated'].keys(), min(log['LogRegError'].values())-0.2, max(log['LogRegError'].values())+0.2, colors='r', linestyles='dashed', label='Model Updated')
    plt.xlabel('Timesteps')
    plt.ylabel('LogRegError')
    plt.hlines(0, min(log['LogRegError'].keys()), max(log['LogRegError'].keys()), colors='black', linestyles='dashed', label='No error')
    plt.legend(loc='upper right')
    plt.xticks(rotation=90)
    plt.show()