import numpy as np
import pandas as pd
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
    
def TimeframeTrigger(model, updateTimestep, dataStart, dataEnd):
    return MethodType(lambda self, x: __TimeframeTrigger(self, x, updateTimestep, dataStart, dataEnd), model)
    
def __TimeframeTrigger(self, input_data, updateTimestep, dataStart, dataEnd):
    """Trigger function to update model based on a fixed time interval.

    Args:
        input_data (dataframe): DataFrame with column of the predicted outcome.
        updateTimestep (pd.Timedelta): Time interval at which to update the model. Note: The model 
        can only be recalibrated at the end of the time interval. If the prediction window is less 
        than the updateTimestep, the model will not be recalibrated.

    Returns:
        bool: Returns True if model update is required.
    """
    try:
        if updateTimestep == 'week':
            self.updateTimestep = pd.Timedelta(weeks=1)
        elif updateTimestep == 'day':
            self.updateTimestep = pd.Timedelta(days=1)
        elif updateTimestep == 'month':
            self.updateTimestep = pd.Timedelta(weeks=4)
        elif isinstance(updateTimestep, int):
            self.updateTimestep = pd.Timedelta(days=updateTimestep)
        else:
            raise TypeError
    except (ValueError, TypeError):
        print("Invalid timestep value, updateTimestep must be 'week', 'day', 'month' or an integer representing days. Defaulting to 'week'.")
        self.updateTimestep = pd.Timedelta(weeks=1)

    update_dates = pd.date_range(start=dataStart, end=dataEnd, freq=self.updateTimestep)

    # Check if current period is in the list of update dates
    if any(date in input_data[self.dateCol].values for date in update_dates):
        return True
    else:
        return False
    
    

    