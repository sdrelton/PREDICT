import numpy as np
from types import MethodType
from PREDICT.Models import PREDICTModel

def AccuracyThreshold(model, threshold, prediction_threshold=0.5):
    return MethodType(lambda self, x: __AccuracyThreshold(self, x, threshold, prediction_threshold), model)

def __AccuracyThreshold(self, input_data, threshold, prediction_threshold):
    """Trigger function to update model if accuracy falls below a given threshold.

    Args:
        input_data (dataframe): DataFrame with column of the predicted outcome.
        threshold (float, optional): Probability threshold at which to classify individuals. Defaults to 0.5.
        prediction_threshold (float): Static accuracy threshold to trigger model update. 

    Returns:
        bool: Returns True if model update is required.
    """
    preds = self.predict(input_data)
    outcomes = input_data[self.outcomeColName].astype(int)
    preds_rounded = np.array(preds >= prediction_threshold).astype(int)
    accuracy = np.mean(preds_rounded == outcomes)
    if accuracy >= threshold:
        return False
    else:
        return True
    
    