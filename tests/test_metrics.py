from PREDICT import Metrics
import numpy as np
import pandas as pd
import warnings

from statsmodels.tools.sm_exceptions import PerfectSeparationWarning


class MockModel:
    def predict(self, df):
        return df['probability']

def test_accuracy_computation():
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

def test_auroc_computation():
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

def test_auprc_computation():
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

def test_f1_computation():
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

def test_precision_computation():
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

def test_recall_computation():
    model = MockModel()
    data = {
        'probability': [0.1, 0.4, 0.6, 0.9],
        'outcome': [0, 0, 1, 1]
    }
    df = pd.DataFrame(data)
    
    recall = Metrics.Recall(model, 'outcome', 0.5)
    hookname, result = recall(df)

    assert hookname == 'Recall'
    assert np.isclose(result, 1.0)

def test_specificity_computation():
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

def test_sensitivity_computation():
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


def test_c_slope_to_r():
    model = MockModel()
    data = {
    'probability': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
    'outcome': [1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0]}
    df = pd.DataFrame(data)
    
    calibration = Metrics.CalibrationSlope(model, 'outcome')
    _, result = calibration(df)

    assert np.isclose(result, 0.0365555720433193, atol=1e-3)

def test_c_slope_to_r2():
    model = MockModel()
    data = {
    'probability': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
    'outcome': [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1]}
    df = pd.DataFrame(data)
    
    calibration = Metrics.CalibrationSlope(model, 'outcome')
    _, result = calibration(df)

    assert np.isclose(result, 0.468914472898616, atol=1e-3)

def test_CITL_calculation():
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
    model = MockModel()
    data = {
        'probability': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
        'outcome': [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1]}
    df = pd.DataFrame(data)
    
    citl = Metrics.CITL(model, 'outcome')
    _, result = citl(df)

    assert np.isclose(result, -0.336888954901901, atol=1e-3)

def test_OE_calculation():
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
    model = MockModel()
    data = {
        'probability': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
        'outcome': [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1]}
    df = pd.DataFrame(data)
    
    oe = Metrics.OE(model, 'outcome')
    _, result = oe(df)

    assert np.isclose(result, 0.875)

def test_coxsnell_R2_calculation():
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