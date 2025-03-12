import pytest
import pandas as pd
import numpy as np
from PREDICT.Models import PREDICTModel
from PREDICT.Triggers import AccuracyThreshold
from sklearn.metrics import accuracy_score

class MockModel:
    """Mock model class for testing"""
    def __init__(self, preds):
        self.preds = preds
        self.outcomeColName = 'outcome'
    
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

