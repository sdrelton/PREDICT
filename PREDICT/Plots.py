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
    # Plot the accuracy in a linegraph
    plt.plot(log['Accuracy'].keys(), log['Accuracy'].values())
    # Add dashed line to indicate when the model was recalibrated
    if 'Model Updated' in log:
        plt.vlines(log['Model Updated'].keys(), 0, 1, colors='r', linestyles='dashed')
    if recalthreshold != None:
        plt.text(min(log['Accuracy'].keys()), recalthreshold+0.01, f'Recalibration Threshold: {recalthreshold*100}%', fontsize=10, color='grey')
    if 'Model Updated' in log:
        plt.legend(['Accuracy', 'Model Updated'], loc='lower right')
    plt.xlabel('Timesteps')
    plt.ylabel('Accuracy')
    plt.show()


def CalibrationSlopePlot(log):
    """Plot the calibration slope of the model over time.

    Args:
        log (dict): Log of model metrics over time and when the model was updated.
    """
    plt.plot(log['CalibrationSlope'].keys(), log['CalibrationSlope'].values())

    plt.axhline(y=1, color='black', linestyle='--')
    plt.annotate('', xy=(max(log['CalibrationSlope'].keys()), 1.35), xytext=(max(log['CalibrationSlope'].keys()), 1.1),
                arrowprops=dict(facecolor='green', shrink=0.05))
    plt.text(min(log['CalibrationSlope'].keys()), 1.1, 'Overestimation', fontsize=10, color='green')
    plt.annotate('', xy=(max(log['CalibrationSlope'].keys()), 0.65), xytext=(max(log['CalibrationSlope'].keys()), 0.9),
                arrowprops=dict(facecolor='red', shrink=0.05))
    plt.text(min(log['CalibrationSlope'].keys()), 0.85, 'Underestimation', fontsize=10, color='red')

    # Add dashed line to indicate when the model was recalibrated
    if 'Model Updated' in log:
        plt.vlines(log['Model Updated'].keys(), 0, 5, colors='r', linestyles='dashed')
    plt.ylim(0, 3)
    plt.xlabel('Timesteps')
    plt.ylabel('Calibration Slope')
    if 'Model Updated' in log:
        plt.legend(['Calibration Slope', 'Model Updated'], loc='upper right')
    else:
        plt.legend(['Calibration Slope'], loc='upper right')
    plt.show()

def CoxSnellPlot(log):
    """Plot the Cox-Snell R^2 of the model over time.

    Args:
        log (dict): Log of model metrics over time and when the model was updated.
    """
    plt.plot(log['CoxSnellR2'].keys(), log['CoxSnellR2'].values())
    if 'Model Updated' in log:
        plt.vlines(log['Model Updated'].keys(), min(log['CoxSnellR2'].values())-0.2, max(log['CoxSnellR2'].values())+0.2, colors='r', linestyles='dashed')
    plt.ylim(min(log['CoxSnellR2'].values())-0.01, max(log['CoxSnellR2'].values())+0.01)
    plt.xlabel('Timesteps')
    plt.ylabel('Cox and Snell R2')
    plt.legend(['Cox and Snell R2', 'Model Updated'], loc='lower left')
    plt.show()

def CITLPlot(log):
    """Plot the Calibration-in-the-Large of the model over time.

    Args:
        log (dict): Log of model metrics over time and when the model was updated.
    """
    plt.figure()
    plt.plot(log['CITL'].keys(), log['CITL'].values())
    plt.axhline(y=0, color='black', linestyle='--')
    plt.annotate('', xy=(max(log['CITL'].keys()), 0.35), xytext=(max(log['CITL'].keys()), 0.1),
                arrowprops=dict(facecolor='green', shrink=0.05))
    plt.text(min(log['CITL'].keys()), 0.3, 'Underestimation', fontsize=10, color='green')
    plt.annotate('', xy=(max(log['CITL'].keys()), -0.3), xytext=(max(log['CITL'].keys()), -0.1),
                arrowprops=dict(facecolor='red', shrink=0.05))
    plt.text(min(log['CITL'].keys()), -0.35, 'Overestimation', fontsize=10, color='red')
    if 'Model Updated' in log:
        plt.vlines(log['Model Updated'].keys(), min(log['CITL'].values())-0.2, max(log['CITL'].values())+0.2, colors='r', linestyles='dashed')
    plt.ylim(min(log['CITL'].values()), max(log['CITL'].values()))
    plt.title('CITL')
    plt.xlabel('Timesteps')
    plt.ylabel('CITL')
    if 'Model Updated' in log:
        plt.legend(['CITL', 'Model Updated'], loc='lower left')
    else:
        plt.legend(['CITL'], loc='lower left')
    plt.show()

def AUROCPlot(log):
    """Plot the AUROC of the model over time.

    Args:
        log (dict): Log of model metrics over time and when the model was updated.
    """
    plt.plot(log['AUROC'].keys(), log['AUROC'].values())
    if 'Model Updated' in log:
        plt.vlines(log['Model Updated'].keys(), min(log['AUROC'].values())-0.2, max(log['AUROC'].values())+0.2, colors='r', linestyles='dashed')
    plt.xlabel('Timesteps')
    plt.ylabel('AUROC')
    if 'Model Updated' in log:
        plt.legend(['AUROC', 'Model Updated'], loc='lower left')
    else:
        plt.legend(['AUROC'], loc='lower left')
    plt.show()

def AUPRCPlot(log):
    """Plot the AUPRC of the model over time.

    Args:
        log (dict): Log of model metrics over time and when the model was updated.
    """
    plt.plot(log['AUPRC'].keys(), log['AUPRC'].values())
    if 'Model Updated' in log:
        plt.vlines(log['Model Updated'].keys(), min(log['AUPRC'].values())-0.2, max(log['AUPRC'].values())+0.2, colors='r', linestyles='dashed')
    plt.xlabel('Timesteps')
    plt.ylabel('AUPRC')
    if 'Model Updated' in log:
        plt.legend(['AUPRC', 'Model Updated'], loc='lower left')
    else:
        plt.legend(['AUPRC'], loc='lower left')
    plt.show()

def F1ScorePlot(log):
    """Plot the F1 Score of the model over time.

    Args:
        log (dict): Log of model metrics over time and when the model was updated.
    """
    plt.plot(log['F1score'].keys(), log['F1score'].values())
    if 'Model Updated' in log:
        plt.vlines(log['Model Updated'].keys(), min(log['F1score'].values())-0.2, max(log['F1score'].values())+0.2, colors='r', linestyles='dashed')
    plt.xlabel('Timesteps')
    plt.ylabel('F1 Score')
    if 'Model Updated' in log:
        plt.legend(['F1 Score', 'Model Updated'], loc='lower left')
    else:
        plt.legend(['F1 Score'], loc='lower left')
    plt.show()

def PrecisionPlot(log):
    """Plot the Precision of the model over time.

    Args:
        log (dict): Log of model metrics over time and when the model was updated.
    """
    plt.plot(log['Precision'].keys(), log['Precision'].values())
    if 'Model Updated' in log:
        plt.vlines(log['Model Updated'].keys(), min(log['Precision'].values())-0.2, max(log['Precision'].values())+0.2, colors='r', linestyles='dashed')
    plt.xlabel('Timesteps')
    plt.ylabel('Precision')
    if 'Model Updated' in log:
        plt.legend(['Precision', 'Model Updated'], loc='lower left')
    else:
        plt.legend(['Precision'], loc='lower left')
    plt.show()

def SensitivityPlot(log):
    """Plot the Sensitivity of the model over time.

    Args:
        log (dict): Log of model metrics over time and when the model was updated.
    """
    plt.plot(log['Sensitivity'].keys(), log['Sensitivity'].values())
    if 'Model Updated' in log:
        plt.vlines(log['Model Updated'].keys(), min(log['Sensitivity'].values())-0.2, max(log['Sensitivity'].values())+0.2, colors='r', linestyles='dashed')
    plt.xlabel('Timesteps')
    plt.ylabel('Sensitivity')
    if 'Model Updated' in log:
        plt.legend(['Sensitivity', 'Model Updated'], loc='lower left')
    else:
        plt.legend(['Sensitivity'], loc='lower left')
    plt.show()

def SpecificityPlot(log):
    """Plot the Specificity of the model over time.

    Args:
        log (dict): Log of model metrics over time and when the model was updated.
    """
    plt.plot(log['Specificity'].keys(), log['Specificity'].values())
    if 'Model Updated' in log:
        plt.vlines(log['Model Updated'].keys(), min(log['Specificity'].values())-0.2, max(log['Specificity'].values())+0.2, colors='r', linestyles='dashed')
    plt.xlabel('Timesteps')
    plt.ylabel('Specificity')
    if 'Model Updated' in log:
        plt.legend(['Specificity', 'Model Updated'], loc='lower left')
    else:
        plt.legend(['Specificity'], loc='lower left')
    plt.show()

def OEPlot(log):
    """Plot the observation to expectation ratio of the model over time.

    Args:
        log (dict): Log of model metrics over time and when the model was updated.
    """
    plt.plot(log['O/E'].keys(), log['O/E'].values())
    if 'Model Updated' in log:
        plt.vlines(log['Model Updated'].keys(), min(log['O/E'].values())-0.2, max(log['O/E'].values())+0.2, colors='r', linestyles='dashed')
    plt.xlabel('Timesteps')
    plt.ylabel('O/E')
    if 'Model Updated' in log:
        plt.legend(['Odds Error', 'Model Updated'], loc='lower left')
    else:
        plt.legend(['Odds Error'], loc='lower left')
    plt.show()
