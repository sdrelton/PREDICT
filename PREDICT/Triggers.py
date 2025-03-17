import numpy as np
import pandas as pd
from types import MethodType
from PREDICT.Models import PREDICTModel
from dateutil.relativedelta import relativedelta

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
    """Create a list of dates to update the model based on a fixed time interval.

    Args:
        model (PREDICTModel): The model to evaluate, must have a predict method.
        updateTimestep (pd.Timedelta): Time interval at which to update the model. Note: The model 
        can only be recalibrated at the end of the time interval. If the prediction window is less 
        than the updateTimestep, the model will not be recalibrated.
        dataStart (pd.Timedelta): Date of when to start regular recalibration.
        dataEnd (pd.Timedelta): Date of when to end regular recalibration.

    Raises:
        TypeError: If updateTimestep is not a valid input.

    Returns:
        tuple: A tuple containing:
        - pd.Timedelta: The calculated time interval for updating the model.
        - pd.DatetimeIndex: A range of dates specifying the update schedule, excluding the first window.
    """

    try:
        if updateTimestep == 'week':
            updateTimestep = pd.Timedelta(weeks=1)
        elif updateTimestep == 'day':
            updateTimestep = pd.Timedelta(days=1)
        elif updateTimestep == 'month':
            #updateTimestep = pd.Timedelta(weeks=4)
            updateTimestep = relativedelta(months=1)
        elif isinstance(updateTimestep, int):
            updateTimestep = pd.Timedelta(days=updateTimestep)
        else:
            raise TypeError
    except (ValueError, TypeError):
        print("Invalid timestep value, updateTimestep must be 'week', 'day', 'month' or an integer representing days. Defaulting to 'week'.")
        updateTimestep = pd.Timedelta(weeks=1)

    # List of dates to update the model excluding the first window
    update_dates = []
    current_date = dataStart + updateTimestep
    while current_date <= dataEnd:
        update_dates.append(current_date)
        current_date += updateTimestep

    return MethodType(lambda self, x: __TimeframeTrigger(self, x, update_dates), model)
    
def __TimeframeTrigger(self, input_data, update_dates):
    """Trigger function to update model based on a fixed time interval.

    Args:
        input_data (dataframe): DataFrame with column of the predicted outcome.
        update_dates (list): List of dates to update the model.

    Returns:
        bool: Returns True if model update is required.
    """

    # Check if current period is in the list of update dates
    if any(date in input_data[self.dateCol].values for date in update_dates):
        return True
    else:
        return False
    
    

    