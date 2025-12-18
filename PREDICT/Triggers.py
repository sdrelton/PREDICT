import numpy as np
import pandas as pd
from types import MethodType
from PREDICT.Models import PREDICTModel, RecalibratePredictions
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LogisticRegression
from scipy.special import logit
from PREDICT import Metrics
import sklearn as skl

def AccuracyThreshold(model, pos_threshold=0.5, update_threshold=0.7):
    return MethodType(lambda self, x: __AccuracyThreshold(self, x, pos_threshold, update_threshold), model)

def __AccuracyThreshold(self, input_data, pos_threshold, update_threshold):
    """Trigger function to update model if accuracy falls below a given threshold.

    Args:
        input_data (dataframe): DataFrame with column of the predicted outcome.
        pos_threshold (float, optional): Probability threshold at which to classify individuals. Defaults to 0.5.
        update_threshold (float): Static accuracy threshold to trigger model update. Defaults to 0.7.

    Returns:
        bool: Returns True if model update is required.
    """
    preds = self.predict(input_data)
    outcomes = input_data[self.outcomeColName].astype(int)
    preds_rounded = np.array(preds >= pos_threshold).astype(int)
    accuracy = np.mean(preds_rounded == outcomes)
    if accuracy >= update_threshold:
        return False
    else:
        return True
    

def AUROCThreshold(model, pos_threshold=0.5, update_threshold=0.7):
    return MethodType(lambda self, x: __AUROCThreshold(self, x, pos_threshold, update_threshold), model)

def __AUROCThreshold(self, input_data, pos_threshold, update_threshold):
    """Trigger function to update model if AUROC falls below a given threshold.

    Args:
        input_data (dataframe): DataFrame with column of the predicted outcome.
        pos_threshold (float, optional): Probability threshold at which to classify individuals. Defaults to 0.5.
        update_threshold (float): Static AUROC threshold to trigger model update. Defaults to 0.7.

    Returns:
        bool: Returns True if model update is required.
    """
    preds = self.predict(input_data)
    outcomes = input_data[self.outcomeColName].astype(int)
    preds_rounded = np.array(preds >= pos_threshold).astype(int)

    fpr, tpr, _ = skl.metrics.roc_curve(outcomes, preds_rounded)
    auroc = skl.metrics.auc(fpr, tpr)

    if auroc >= update_threshold:
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

    if updateTimestep == 'week':
        freq = pd.Timedelta(weeks=1)
    elif updateTimestep == 'day':
        freq = pd.Timedelta(days=1)
    elif updateTimestep == 'month':
        freq = pd.DateOffset(months=1)
    elif isinstance(updateTimestep, int):
        freq = pd.Timedelta(days=updateTimestep)
    else:
        freq = pd.Timedelta(weeks=1)
        raise TypeError("Invalid timestep value, updateTimestep must be 'week', 'day', 'month' or an integer representing days. Defaulting to 'week'.")
        

    update_dates = pd.date_range(start=dataStart + freq, end=dataEnd, freq=freq)


    return MethodType(lambda self, x: __TimeframeTrigger(self, x, update_dates), model)
    
def __TimeframeTrigger(self, input_data, update_dates, tolerance=pd.Timedelta(days=5)):
    """Trigger function to update model based on a fixed time interval.

    Args:
        input_data (dataframe): DataFrame with column of the predicted outcome.
        update_dates (list): List of dates to update the model.

    Returns:
        bool: Returns True if model update is required.
    """

    current_date = input_data[self.dateCol].max()
    return any(abs(current_date - d) <= tolerance for d in update_dates)
    
    
def SPCTrigger(model, input_data, dateCol='date', clStartDate=None, clEndDate=None, 
            numMonths=None, warningCL=None, recalCL=None, warningSDs=2, recalSDs=3, 
            verbose=True):
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
        input_data (dataframe): DataFrame with column of the predicted outcome.
        dateCol (str): Column containing the dates.
        clStartDate (str): Start date to determine control limits from. Defaults to None.
        clEndDate (str): End date to determine control limits from. Defaults to None.
        numMonths (int): The number of months to base the control limits on. Defaults to None.
        warningCL (float): A manually set control limit for the warning control limit. Defaults to None.
        recalCL (float): A manually set control limit for the recalibration trigger limit. Defaults to None.
        warningSDs (int or float): Number of standard deviations from the mean to set the warning limit to. Defaults to 2.
        recalSDs (int or float): Number of standard deviations from the mean to set the recalibration trigger to. Defaults to 3.
        verbose (bool): If True, prints the control limit warnings. Defaults to True.

    Returns:
        tuple: A tuple containing:
        - pd.Timedelta: The calculated time interval for updating the model.
        - pd.DatetimeIndex: A range of dates specifying the update schedule.
    """
    if warningSDs > recalSDs:
        raise ValueError(f"warningSDs must be lower than recalSDs. recalSDs {recalSDs} is > warningSDs {warningSDs}")
        
    if clStartDate and clEndDate and not any([numMonths, warningCL, recalCL]):
        startCLDate = pd.to_datetime(clStartDate, dayfirst=True)
        endCLDate = pd.to_datetime(clEndDate, dayfirst=True)

    elif numMonths and not any([clStartDate, clEndDate, warningCL, recalCL]):
        startCLDate = input_data[dateCol].min()
        endCLDate = startCLDate + relativedelta(months=numMonths)

    elif warningCL and recalCL and not any([numMonths, clStartDate, clEndDate]):
        if warningCL > recalCL:
            raise ValueError(f"Warning control limit must be lower than the recalibration control limit. recalCL {recalCL} is > warningCL {warningCL}")
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
            warning zone (warningSDs) and the start of the recalibration zone (recalSDs), 
            alongside either numMonths or clStartDate and clEndDate.""")


    u2sdl, u3sdl, l2sdl, l3sdl = model.CalculateControlLimits(input_data, startCLDate, endCLDate, warningCL, recalCL, warningSDs, recalSDs)

    return MethodType(lambda self, x: __SPCTrigger(self, x, model, u2sdl, u3sdl, l2sdl, l3sdl, verbose), model)

def __SPCTrigger(self, input_data, model, u2sdl, u3sdl, l2sdl, l3sdl, verbose):
    """Trigger function to determine whether recalibration should be carried out and whether a warning message should be 
    displayed prompting the user to investigate increasing errors in the model.

    Args:
        input_data (dataframe): DataFrame with column of the predicted outcome.
        model (PREDICTModel): The model to evaluate, must have a predict method.
        u2sdl (float): First upper control limit above the mean.
        u3sdl (float): Uppermost control limit above the mean.
        l2sdl (float): First lower control limit beneath the mean.
        l3sdl (float): Lowest control limit beneath the mean.

    Returns:
        bool: True to trigger model recalibration.
    """

    _, error = Metrics.__SumOfDiffComputation(model, input_data, self.outcomeColName)
    # if error enter yellow zone (usually between 2SD and 3SD unless user has manually changed control limits) then print warning message
    if error > u2sdl and error < u3sdl:
        curDate = input_data[self.dateCol].max()
        if verbose:
            print(f'{curDate}: Error is in the upper warning zone. \nInvestigate the cause of the increase in error.\n')
        return False
    # if error enter red zone (usally above 3SD) then recalibrate the model
    elif error > u3sdl:
        curDate = input_data[self.dateCol].max()
        if verbose:
            print(f'{curDate}: Error is in the upper danger zone. \nThe model has been recalibrated. \nYou might want to investigate the cause of the error increasing.\n')
        return True
    
    elif error < l2sdl and error > l3sdl:
        curDate = input_data[self.dateCol].max()
        if verbose:
            print(f'{curDate}: Error is in the lower warning zone. \nInvestigate the cause of the increase in error.\n')
        return False

    elif error < l3sdl:
        curDate = input_data[self.dateCol].max()
        if verbose:
            print(f'{curDate}: Error is in the lower danger zone. \nThe model has been recalibrated. \nYou might want to investigate the cause of the error increasing.\n')
        return True
    
    else:
        return False