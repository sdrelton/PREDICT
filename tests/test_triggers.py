import pytest
import pandas as pd
import numpy as np
from PREDICT import PREDICT
from PREDICT.Models import PREDICTModel, RecalibratePredictions
from PREDICT.Triggers import AccuracyThreshold, SPCTrigger, BayesianRefitTrigger
from sklearn.metrics import accuracy_score
from datetime import datetime

class MockModel:
    """Mock model class for testing"""
    def __init__(self, preds):
        self.preds = preds
        self.outcomeColName = 'outcome'
        self.dateCol = "date"
    
    def predict(self, input_data):
        return self.preds

@pytest.fixture
def input_data():
    """Create dummy binary outcome data for testing.

    Returns:
        pd.DataFrame: Dummy binary outcome data.
    """
    return pd.DataFrame({
        'outcome': [0, 1, 0, 1, 1]
    })

def test_accuracy_above_threshold(input_data):
    """Test that the accuracy threshold trigger works when accuracy is above the threshold.

    Args:
        input_data (pd.DataFrame): Dummy binary outcome data.
    """
    probs = np.array([0.6, 0.7, 0.4, 0.8, 0.9])
    model = MockModel(preds=probs)
    threshold = 0.5
    prediction_threshold = 0.5
    preds = np.where(probs > prediction_threshold, 1, 0)
    print("Accuracy: ", accuracy_score(input_data[model.outcomeColName], preds))
    model.trigger = AccuracyThreshold(model, threshold, prediction_threshold)
    bool_out = model.trigger(input_data)
    assert bool_out == False, "Accuracy is above threshold, no update required."

def test_accuracy_below_threshold(input_data):
    """Test that the accuracy threshold trigger works when accuracy is below the threshold.

    Args:
        input_data (pd.DataFrame): Dummy binary outcome data.
    """
    probs = np.array([0.4, 0.5, 0.9, 0.4, 0.4])
    model = MockModel(preds=probs)
    threshold = 0.7
    prediction_threshold = 0.5
    preds = np.where(probs > prediction_threshold, 1, 0)
    print("Accuracy: ", accuracy_score(input_data[model.outcomeColName], preds))
    model.trigger = AccuracyThreshold(model, threshold, prediction_threshold)
    bool_out = model.trigger(input_data)
    assert bool_out == True, "Accuracy is below threshold, update required."

def test_accuracy_same_as_threshold(input_data):
    """Test that the accuracy threshold trigger works when accuracy is equal to the threshold.

    Args:
        input_data (pd.DataFrame): Dummy binary outcome data.
    """
    probs = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    model = MockModel(preds=probs)
    threshold = 0.5
    prediction_threshold = 0.5
    preds = np.where(probs > prediction_threshold, 1, 0)
    print("Accuracy: ", accuracy_score(input_data[model.outcomeColName], preds))
    model.trigger = AccuracyThreshold(model, threshold, prediction_threshold)
    bool_out = model.trigger(input_data)
    assert bool_out == False, "Accuracy is at threshold, no update required."

def test_control_limit_input():
    probs = np.array([0.4, 0.5, 0.9, 0.4, 0.4])
    model = MockModel(preds=probs)
    with pytest.raises(ValueError):
        model.trigger = SPCTrigger(model=model, input_data=input_data, warningSDs=3, recalSDs=4, recalCL=0.5)
    with pytest.raises(ValueError):
        model.trigger = SPCTrigger(model=model, input_data=input_data, warningSDs=3, clEndDate='31-01-2025')
    with pytest.raises(ValueError):
        model.trigger = SPCTrigger(model=model, input_data=input_data, recalCL=0.5, numMonths=6)

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "date": pd.date_range(start="2023-01-01", periods=13, freq="ME")
    })


def test_bayesian_refit_trigger(sample_data):
    """
    Test the Bayesian refit trigger function.

    Args:
        sample_data (DataFrame): Sample datetime data for testing.
    """
    probs = np.array([0.4, 0.5, 0.9, 0.4, 0.4])
    model = MockModel(probs)
    refit_func = BayesianRefitTrigger(model, sample_data, refitFrequency=6)

    # Check if the function is bound correctly
    assert hasattr(refit_func, "__call__"), "The returned function should be callable."

    # Create sample inputs to validate refit triggering
    trigger_dates = [
        datetime(2023, 7, 31),  # Expected refit date after 6 months
        datetime(2024, 1, 31),  # Expected refit date after another 6 months
    ]
    
    for date in trigger_dates:
        test_data = pd.DataFrame({"date": [date]})
        assert refit_func(test_data) is True, f"Refit should trigger on {date}"

    # Test a non-triggering date
    test_data = pd.DataFrame({"date": [datetime(2023, 3, 31)]})
    assert refit_func(test_data) is False, "Refit should NOT trigger on a non-refit date."
