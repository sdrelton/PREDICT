import numpy as np
import pandas as pd
from types import MethodType
from PREDICT.Models import PREDICTModel, RecalibratePredictions
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LogisticRegression
from scipy.special import logit
from PREDICT import Metrics

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
    
    
def SPCTrigger(model, input_data, dateCol='date', clStartDate=None, clEndDate=None, 
            numMonths=None, warningCL=None, recalCL=None, warningSDs=None, recalSDs=None):
    """Trigger function to update the model if the error enters an upper control limit.
        The control limits can be set using one of the following methods:
        - Enter a start (clStartDate) and end date (clEndDate) to determine the control 
            limits using the error mean and std during this period.
        - Enter the number of months (numMonths) to base the control limits on from the start of the period.
        - Manually set the control limits by entering the float values for the 'warning' (warningCL) and 'recalibration' (recalCL) zones.
        - Enter the number of standard deviations from the mean for the start of the warning zone (warningSDs) and the start of the 
            recalibration zone (recalSDs).

    Args:
        model (PREDICTModel): The model to evaluate, must have a predict method.
        clStartDate (str): Start date to determine control limits from. Defaults to None.
        clEndDate (str): End date to determine control limits from. Defaults to None.
        numMonths (int): The number of months to base the control limits on. Defaults to None.
        warningCL (float): A manually set control limit for the warning control limit.
        recalCL (float): A manually set control limit for the recalibration trigger limit.
        warningSDs (int or float): Number of standard deviations from the mean to set the warning limit to.
        recalSDs (int or float): Number of standard deviations from the mean to set the recalibration trigger to.

    Returns:
        tuple: A tuple containing:
        - pd.Timedelta: The calculated time interval for updating the model.
        - pd.DatetimeIndex: A range of dates specifying the update schedule.
    """

    if clStartDate is not None and clEndDate is not None and numMonths is None and warningCL is None and recalCL is None and warningSDs is None and recalSDs is None:
        startCLDate = pd.to_datetime(clStartDate, dayfirst=True)
        endCLDate = pd.to_datetime(clEndDate, dayfirst=True)
    # Use the first X months of data to determine the control limits
    elif numMonths is not None and clStartDate is None and clEndDate is None and warningCL is None and recalCL is None and warningSDs is None and recalSDs is None:
        startCLDate = input_data[dateCol].min() 
        endCLDate = startCLDate + relativedelta(months=numMonths)
    elif warningCL is not None and recalCL is not None and numMonths is None and clStartDate is None and clEndDate is None and warningSDs is None and recalSDs is None:
        if warningCL > recalCL:
            raise ValueError("Warning control limit must be lower than the recalibration control limit.")
        startCLDate = None
        endCLDate = None
    elif warningSDs is not None and recalSDs is not None and numMonths is None and clStartDate is None and clEndDate is None and warningCL is None and recalCL is None:
        if warningSDs > recalSDs:
            raise ValueError("Warning control limit must be lower than the recalibration control limit (warningSDs > recalSDs). ")
        startCLDate = None
        endCLDate = None
    
    else:
        raise ValueError("""The control limits must be set using ONE of the following methods:\n
        - Enter a start (clStartDate) and end date (clEndDate) to determine the control 
            limits using the error mean and std during this period. \n
        - Enter the number of months (numMonths) to base the control limits on from the 
            start of the period.\n
        - Manually set the control limits by entering the float values for the 'warning' 
            (warningCL) and 'recalibration' (recalCL) zones.\n
        - Enter the number of standard deviations from the mean for the start of the 
            warning zone (warningSDs) and the start of the recalibration zone (recalSDs).""")


    u2sdl, u3sdl= model.CalculateControlLimits(input_data, startCLDate, endCLDate, warningCL, recalCL, warningSDs, recalSDs)

    return MethodType(lambda self, x: __SPCTrigger(self, x, model, u2sdl, u3sdl), model)

def __SPCTrigger(self, input_data, model, u2sdl, u3sdl):

    _, error = Metrics.__LogRegErrorComputation(model, input_data, self.outcomeColName)
    # if error enter yellow zone (between 2SD and 3SD) then print warning message
    if error > u2sdl and error < u3sdl:
        curDate = input_data[self.dateCol].max()
        print(f'{curDate}: Error is in the warning zone (+2 standard deviations from the mean). \nInvestigate the cause of the increase in error.\n')
        return False
    # if error enter red zone (above 3SD) then recalibrate the model
    elif error > u3sdl:
        curDate = input_data[self.dateCol].max()
        print(f'{curDate}: Error is in the danger zone (+3 standard deviations from the mean). \nThe model has been recalibrated. \nYou might want to investigate the cause of the error increasing.\n')
        return True
    
    else:
        return False
    