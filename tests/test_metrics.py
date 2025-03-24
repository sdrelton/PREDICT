import os
from PREDICT import Metrics
import numpy as np
import pandas as pd
import warnings
from unittest.mock import MagicMock
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

# Check if the code is being executed by Sphinx
if any(env in os.getenv("READTHEDOCS", "").lower() or os.getenv("SPHINX_BUILD", "").lower() for env in ["true", "sphinx"]):
    # Mock the pd.read_csv function to prevent Sphinx from crashing
    pd.read_csv = MagicMock(return_value=pd.DataFrame({
        'probability': [0.1, 0.4, 0.6, 0.9],
        'outcome': [0, 0, 1, 1]
    }))

# Use the mocked or real pd.read_csv depending on the environment
try:
    hd_outcomes_df = pd.read_csv('tests/hd_model_predictions.csv')
except FileNotFoundError:
    # Fallback for environments where the file is not available
    hd_outcomes_df = pd.DataFrame({
        'probability': [0.1, 0.4, 0.6, 0.9],
        'outcome': [0, 0, 1, 1]
    })

class MockModel:
    """Mock model class for testing.
    """
    def predict(self, df):
        return df['probability']

def test_accuracy_computation():
    """Tests the computation of the accuracy metric 
    on dummy data assuming a perfect model.
    """
    model = MockModel()
    data = {
        'probability': [0.1, 0.4, 0.6, 0.9],
        'outcome': [0, 0, 1, 1]
    }
    df = pd.DataFrame(data)
    
    accuracy = Metrics.Accuracy(model, 'outcome', 0.5)
    hookname, result = accuracy(df)

    assert hookname == 'Accuracy'
    assert result == 1.0

def test_accuracy_computation2():
    """Tests the computation of the accuracy metric 
    on a model trained on the heart disease dataset.
    """
    model = MockModel()
    
    accuracy = Metrics.Accuracy(model, 'outcome', 0.5)
    _, result = accuracy(hd_outcomes_df)

    assert round(result,2) == 0.88

def test_auroc_computation():
    """Tests the computation of the AUROC metric 
    on dummy data assuming a perfect model.
    """
    model = MockModel()
    data = {
        'probability': [0.1, 0.4, 0.6, 0.9],
        'outcome': [0, 0, 1, 1]
    }
    df = pd.DataFrame(data)
    
    auc = Metrics.AUROC(model, 'outcome')
    hookname, result = auc(df)

    assert hookname == 'AUROC'
    assert np.isclose(result, 1.0)

def test_auroc_computation2():
    """Tests the computation of the AUROC metric 
        on a model trained on the heart disease dataset.
    """
    model = MockModel()
    
    auc = Metrics.AUROC(model, 'outcome')
    _, result = auc(hd_outcomes_df)

    assert round(result,2) == 0.95

def test_auprc_computation():
    """Tests the computation of the AUPRC 
        metric on dummy data assuming a perfect model.
    """
    model = MockModel()
    data = {
        'probability': [0.1, 0.4, 0.6, 0.9],
        'outcome': [0, 0, 1, 1]
    }
    df = pd.DataFrame(data)
    
    auprc = Metrics.AUPRC(model, 'outcome')
    hookname, result = auprc(df)

    assert hookname == 'AUPRC'
    assert np.isclose(result, 1.0)

def test_auprc_computation2():
    """Tests the computation of the AUPRC metric
        on a model trained on the heart disease dataset.
    """
    model = MockModel()
    
    auprc = Metrics.AUPRC(model, 'outcome')
    _, result = auprc(hd_outcomes_df)

    assert round(result,2) == 0.94

def test_f1_computation():
    """Tests the computation of the F1 score metric 
        on dummy data assuming a perfect model.
    """
    model = MockModel()
    data = {
        'probability': [0.1, 0.4, 0.6, 0.9],
        'outcome': [0, 0, 1, 1]
    }
    df = pd.DataFrame(data)
    
    f1 = Metrics.F1Score(model, 'outcome', 0.5)
    hookname, result = f1(df)

    assert hookname == 'F1score'
    assert np.isclose(result, 1.0)

def test_f1_computation2():
    """Tests the computation of the F1 score metric
        on a model trained on the heart disease dataset.
    """
    model = MockModel()
    
    f1 = Metrics.F1Score(model, 'outcome', 0.5)
    _, result = f1(hd_outcomes_df)

    assert round(result,2) == 0.86

def test_precision_computation():
    """Tests the computation of the precision metric 
        on dummy data assuming a perfect model.
    """
    model = MockModel()
    data = {
        'probability': [0.1, 0.4, 0.6, 0.9],
        'outcome': [0, 0, 1, 1]
    }
    df = pd.DataFrame(data)
    
    precision = Metrics.Precision(model, 'outcome', 0.5)
    hookname, result = precision(df)

    assert hookname == 'Precision'
    assert np.isclose(result, 1.0)

def test_precision_computation2():
    """Tests the computation of the precision metric 
        on a model trained on the heart disease dataset.
    """
    model = MockModel()
    
    precision = Metrics.Precision(model, 'outcome', 0.5)
    _, result = precision(hd_outcomes_df)

    assert round(result,2) == 0.89

def test_recall_computation():
    """Tests the computation of the recall metric on 
        dummy data assuming a perfect model.
    """
    model = MockModel()
    data = {
        'probability': [0.1, 0.4, 0.6, 0.9],
        'outcome': [0, 0, 1, 1]
    }
    df = pd.DataFrame(data)
    
    recall = Metrics.Recall(model, 'outcome', 0.5)
    hookname, result = recall(df)

    assert hookname == 'Sensitivity'
    assert np.isclose(result, 1.0)

def test_recall_computation2():
    """Tests the computation of the recall metric 
        on a model trained on the heart disease dataset.
    """
    model = MockModel()
    
    recall = Metrics.Recall(model, 'outcome', 0.5)
    _, result = recall(hd_outcomes_df)

    assert round(result,2) == 0.83

def test_specificity_computation():
    """Tests the computation of the specificity metric 
        on dummy data assuming a perfect model.
    """
    model = MockModel()
    data = {
        'probability': [0.1, 0.4, 0.6, 0.9],
        'outcome': [0, 0, 1, 1]
    }
    df = pd.DataFrame(data)
    
    specificity = Metrics.Specificity(model, 'outcome', 0.5)
    hookname, result = specificity(df)

    assert hookname == 'Specificity'
    assert np.isclose(result, 1.0)

def test_specificity_computation2():
    """Tests the computation of the specificity metric 
        on a model trained on the heart disease dataset.
    """
    model = MockModel()
    
    specificity = Metrics.Specificity(model, 'outcome', 0.5)
    _, result = specificity(hd_outcomes_df)

    assert round(result,2) == 0.92

def test_sensitivity_computation():
    """Tests the computation of the sensitivity metric on 
        dummy data assuming a perfect model.
    """
    model = MockModel()
    data = {
        'probability': [0.1, 0.4, 0.6, 0.9],
        'outcome': [0, 0, 1, 1]
    }
    df = pd.DataFrame(data)
    
    sensitivity = Metrics.Sensitivity(model, 'outcome', 0.5)
    hookname, result = sensitivity(df)

    assert hookname == 'Sensitivity'
    assert np.isclose(result, 1.0)

def test_sensitivity_computation2():
    """Tests the computation of the sensitivity metric 
        on a model trained on the heart disease dataset.
    """
    model = MockModel()
    
    sensitivity = Metrics.Sensitivity(model, 'outcome', 0.5)
    _, result = sensitivity(hd_outcomes_df)

    assert round(result,2) == 0.83


def test_c_slope_to_r():
    """Tests the computation of the calibration slope metric on dummy data
        compared to the results on the same data using R.
    """
    model = MockModel()
    data = {
    'probability': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
    'outcome': [1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0]}
    df = pd.DataFrame(data)
    
    calibration = Metrics.CalibrationSlope(model, 'outcome')
    _, result = calibration(df)

    assert np.isclose(result, 0.0365555720433193, atol=1e-3)

def test_c_slope_to_r2():
    """Tests the computation of the calibration slope metric 
        on dummy data compared to a result from R.
    """
    model = MockModel()
    data = {
    'probability': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
    'outcome': [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1]}
    df = pd.DataFrame(data)
    
    calibration = Metrics.CalibrationSlope(model, 'outcome')
    _, result = calibration(df)

    assert np.isclose(result, 0.468914472898616, atol=1e-3)

def test_c_slope():
    """Tests the computation of the calibration slope metric on a 
        model trained on the heart disease dataset.
    """
    model = MockModel()
    
    calibration = Metrics.CalibrationSlope(model, 'outcome')
    _, result = calibration(hd_outcomes_df)

    assert round(result,2) == 1.36

def test_CITL_calculation():
    """Tests the computation of the calibration-in-the-large metric 
        on dummy data of a perfect model.
    """
    model = MockModel()
    data = {
        'probability': [0.1, 0.4, 0.6, 0.9],
        'outcome': [0, 0, 1, 1]
    }
    df = pd.DataFrame(data)
    
    citl = Metrics.CITL(model, 'outcome')
    hookname, result = citl(df)

    assert hookname == 'CITL'
    assert np.isclose(result, 0.0)

def test_CITL_calculation2():
    """Tests the computation of the calibration-in-the-large metric on 
        dummy data compared to a result from R.
    """
    model = MockModel()
    data = {
        'probability': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
        'outcome': [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1]}
    df = pd.DataFrame(data)
    
    citl = Metrics.CITL(model, 'outcome')
    _, result = citl(df)

    assert np.isclose(result, -0.336888954901901, atol=1e-3)

def test_CITL_calculation3():
    """Tests the computation of the calibration-in-the-large metric 
        on a model trained on the heart disease dataset.
    """
    model = MockModel()
    
    citl = Metrics.CITL(model, 'outcome')
    _, result = citl(hd_outcomes_df)

    assert np.isclose(result, 0.27, atol=1e-3), f"Expected -0.61, got {result}"

def test_OE_calculation():
    """Tests the computation of the O/E metric for a perfect model on dummy data.
    """
    model = MockModel()
    data = {
        'probability': [0.1, 0.4, 0.6, 0.9],
        'outcome': [0, 0, 1, 1]
    }
    df = pd.DataFrame(data)
    
    oe = Metrics.OE(model, 'outcome')
    hookname, result = oe(df)

    assert hookname == 'O/E'
    assert np.isclose(result, 1.0)

def test_OE_calculation2():
    """Tests the computation of the O/E metric on dummy data with the result compared to an R result.
    """
    model = MockModel()
    data = {
        'probability': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
        'outcome': [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1]}
    df = pd.DataFrame(data)
    
    oe = Metrics.OE(model, 'outcome')
    _, result = oe(df)

    assert np.isclose(result, 0.875)

def test_OE_calculation3():
    """Tests the computation of the O/E metric on a model trained on the heart disease dataset.
    """
    model = MockModel()
    
    oe = Metrics.OE(model, 'outcome')
    _, result = oe(hd_outcomes_df)

    assert round(result,2) == 1.08

def test_coxsnell_R2_calculation():
    """Tests the computation of the Cox-Snell R^2 metric for an imperfect 
    model and a model predicting all the same probabilities on dummy data.
    """
    # A perfect model raises a a PerfectSeparationWarning due to LogReg overfitting.
    
    # Imperfect model
    model_imperfect = MockModel()
    df_imperfect = pd.DataFrame({
        'probability': [0.2, 0.3, 0.5, 0.7],
        'outcome': [0, 0, 1, 1]
    })
    coxsnell_imperfect = Metrics.CoxSnellR2(model_imperfect, 'outcome')
    hookname_imperfect, result_imperfect = coxsnell_imperfect(df_imperfect)
    assert hookname_imperfect == 'CoxSnellR2'
    assert result_imperfect < 1.0  # Should be less than 1.0 for imperfect models

    # Edge case - all predictions the same
    model_edge = MockModel()
    df_edge = pd.DataFrame({
        'probability': [0.5, 0.5, 0.5, 0.5],
        'outcome': [0, 0, 1, 1]
    })
    coxsnell_edge = Metrics.CoxSnellR2(model_edge, 'outcome')
    hookname_edge, result_edge = coxsnell_edge(df_edge)
    assert hookname_edge == 'CoxSnellR2'
    assert result_edge == 0.0  # Should be 0 for no variability in predictions

def test_coxsnell_R2_calculation2():
    """Tests the computation of the Cox-Snell R^2 metric 
    on a model trained on the heart disease dataset."""
    model = MockModel()
    
    coxsnell = Metrics.CoxSnellR2(model, 'outcome')
    _, result = coxsnell(hd_outcomes_df)

    assert np.isclose(result, 0.53, atol=1e-3), f"Expected 0.53, got {result}"

def test_sum_of_diff_calculation():
    """Tests the computation of the logistic regression error metric 
    on dummy data assuming a perfect model.
    """
    model = MockModel()
    data = {
        'probability': [0.1, 0.4, 0.6, 0.9],
        'outcome': [0, 0, 1, 1]
    }
    df = pd.DataFrame(data)
    
    SumOfDiff = Metrics.SumOfDiff(model, 'outcome')
    hookname, result = SumOfDiff(df)

    assert hookname == 'SumOfDifferences'
    assert np.isclose(result, 0.0)

def test_sum_of_diff_calculation2():
    """Tests the computation of the logistic regression error metric 
    on dummy data assuming a model that gets predictions right
    25% of the time.
    """
    model = MockModel()
    data = {
        'probability': [0.1, 0.4, 0.6, 0.9],
        'outcome': [0, 1, 0, 0]
    }
    df = pd.DataFrame(data)
    
    SumOfDiff = Metrics.SumOfDiff(model, 'outcome')
    _, result = SumOfDiff(df)

    assert np.isclose(result, -0.25)

def test_sum_of_diff_calculation3():
    """Tests the computation of the logistic regression error metric 
    on dummy data assuming a model that never gets predictions right.
    """
    model = MockModel()
    data = {
        'probability': [0.8, 0.6, 0.1, 0.4],
        'outcome': [0, 0, 1, 1]
    }
    df = pd.DataFrame(data)
    
    SumOfDiff = Metrics.SumOfDiff(model, 'outcome')
    _, result = SumOfDiff(df)

    assert np.isclose(result, 0.025)