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
    
def AlwaysTrigger(model):
    return MethodType(lambda self, x: True, model)
    

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
    auroc = skl.metrics.roc_auc_score(outcomes, preds)
    
    if auroc >= update_threshold:
        return False
    else:
        return True
    
    

def CalibrationSlopeThreshold(model, lower_limit=None, upper_limit=None):
    """Create a trigger that fires when calibration slope falls outside provided bounds.

    Args:
        model (PREDICTModel): The model to evaluate, must have a predict method.
        lower_limit (float, optional): Lower acceptable bound for calibration slope. If None, only upper_limit is used.
        upper_limit (float, optional): Upper acceptable bound for calibration slope. If None, only lower_limit is used.

    Returns:
        MethodType: Bound method that accepts (self, input_data) and returns True when calibration slope is outside bounds.
    """
    return MethodType(lambda self, x: __CalibrationSlopeThreshold(self, x, lower_limit, upper_limit), model)


def __CalibrationSlopeThreshold(self, input_data, lower_limit, upper_limit):
    """Trigger implementation that computes calibration slope for the provided input_data and compares to limits.

    Uses the shared Metrics.CalibrationSlope hook to ensure consistent calculation across the project.

    Returns:
        bool: True if calibration slope is outside [lower_limit, upper_limit] (i.e. requires update), False otherwise.
    """
    if lower_limit is None and upper_limit is None:
        raise ValueError("At least one of lower_limit or upper_limit must be provided for CalibrationSlopeThreshold.")

    # Use the Metrics.CalibrationSlope loghook to compute the calibration slope value
    try:
        _, slope_value = Metrics.CalibrationSlope(self)(input_data)
    except Exception as e:
        # If computation fails, be conservative and trigger an update
        if hasattr(self, 'verbose') and self.verbose:
            print(f"Calibration slope computation failed: {e}. Triggering update by default.")
        return True

    # Check bounds
    if (lower_limit is not None) and (slope_value < lower_limit):
        return True
    if (upper_limit is not None) and (slope_value > upper_limit):
        return True

    return False

def CITLThreshold(model, lower_limit=None, upper_limit=None):
    """Create a trigger that fires when Calibration-In-The-Large (CITL) falls outside provided bounds.

    Args:
        model (PREDICTModel): The model to evaluate, must have a predict method.
        lower_limit (float, optional): Lower acceptable bound for CITL. If None, only upper_limit is used.
        upper_limit (float, optional): Upper acceptable bound for CITL. If None, only lower_limit is used.

    Returns:
        MethodType: Bound method that accepts (self, input_data) and returns True when CITL is outside bounds.
    """
    return MethodType(lambda self, x: __CITLThreshold(self, x, lower_limit, upper_limit), model)


def __CITLThreshold(self, input_data, lower_limit, upper_limit):
    """Trigger implementation that computes CITL for the provided input_data and compares to limits.

    The CITL is computed using the shared Metrics.CITL hook to ensure consistent calculation across the project.

    Returns:
        bool: True if CITL is outside [lower_limit, upper_limit] (i.e. requires update), False otherwise.
    """
    if lower_limit is None and upper_limit is None:
        raise ValueError("At least one of lower_limit or upper_limit must be provided for CITLThreshold.")

    # Use the Metrics.CITL loghook to compute the CITL value
    try:
        _, citl_value = Metrics.CITL(self)(input_data)
    except Exception as e:
        # If computation fails, be conservative and trigger an update
        if hasattr(self, 'verbose') and self.verbose:
            print(f"CITL computation failed: {e}. Triggering update by default.")
        return True

    # Check bounds
    if (lower_limit is not None) and (citl_value < lower_limit):
        return True
    if (upper_limit is not None) and (citl_value > upper_limit):
        return True

    return False

def F1Threshold(model, pos_threshold=0.5, update_threshold=0.7):
    return MethodType(lambda self, x: __F1Threshold(self, x, pos_threshold, update_threshold), model)

def __F1Threshold(self, input_data, pos_threshold, update_threshold):
    """Trigger function to update model if F1 score falls below a given threshold.

    Args:
        input_data (dataframe): DataFrame with column of the predicted outcome.
        pos_threshold (float, optional): Probability threshold at which to classify individuals. Defaults to 0.5.
        update_threshold (float): Static F1 threshold to trigger model update. Defaults to 0.7.

    Returns:
        bool: Returns True if model update is required.
    """
    preds = self.predict(input_data)
    outcomes = input_data[self.outcomeColName].astype(int)
    preds_rounded = np.array(preds >= pos_threshold).astype(int)

    f1 = skl.metrics.f1_score(outcomes, preds_rounded)

    if f1 >= update_threshold:
        return False
    else:
        return True
    


def OEThreshold(model, lower_threshold=0.9, upper_threshold=1.1):
    return MethodType(lambda self, x: __OEThreshold(self, x, lower_threshold, upper_threshold), model)


def __OEThreshold(self, input_data, lower_threshold=0.9, upper_threshold=1.1):
    """Trigger function to update model if O/E falls above or below a given threshold.

    Args:
        input_data (dataframe): DataFrame with column of the predicted outcome.
        lower_threshold (float, optional): Lower threshold for O/E. Defaults to 0.9.
        upper_threshold (float, optional): Upper threshold for O/E. Defaults to 1.1.

    Returns:
        bool: Returns True if model update is required.
    """
    
    predictions = self.predict(input_data)
    mean_pred = predictions.mean()
    mean_outcome = input_data[self.outcomeColName].mean()
    oe_value = mean_outcome / mean_pred if mean_pred != 0 else float('inf')

    if (oe_value <= lower_threshold) or (oe_value >= upper_threshold):
        return True
    else:
        return False

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
    
def __SPCCalculateControlLimits(model, input_data, startCLDate, endCLDate, warningCL, recalCL, warningSDs, recalSDs):
        """Calculate the static control limits of data using either the specific period (startCLDate to endCLDate) or 
        the inputted control limits (warningCL, recalCL).

        Args:
            input_data (pd.DataFrame): The input data for updating the model.
            startCLDate (str): Start date to determine control limits from.
            endCLDate (str): End date to determine control limits from.
            warningCL (float): A manually set control limit for the warning control limit.
            recalCL (float): A manually set control limit for the recalibration trigger limit.
            warningSDs (int or float): Number of standard deviations from the mean to set the warning limit to
            recalSDs (int or float): Number of standard deviations from the mean to set the recalibration trigger to.

        Returns:
            float, float: Two upper control limits for the warning and danger/recalibration trigger zones.
        """
        def CalculateError(group):
            predictions = group['predictions']
            differences = group[model.outcomeColName] - predictions
            sum_of_differences = np.sum(differences)/len(group[model.outcomeColName])
            return sum_of_differences
        
        if startCLDate is not None and endCLDate is not None:
            # Get the logreg error at each timestep within the control limit determination period
            createCLdf = input_data[(input_data[model.dateCol] >= startCLDate) & (input_data[model.dateCol] <= endCLDate)].copy()
            
            # Predictions column
            createCLdf['predictions'] = model.predict(createCLdf)

            errors_by_date = createCLdf.groupby(model.dateCol).apply(CalculateError)
            model.mean_error = errors_by_date.mean()
            std_dev_error = errors_by_date.std() / np.sqrt(len(errors_by_date))
            
            model.u2sdl = model.mean_error + warningSDs * std_dev_error
            model.u3sdl = model.mean_error + recalSDs * std_dev_error
            model.l2sdl = model.mean_error - warningSDs * std_dev_error
            model.l3sdl = model.mean_error - recalSDs * std_dev_error

        #elif warningCL is not None and recalCL is not None:
        else:
            # Predictions column
            input_data['predictions'] = model.predict(input_data)
            errors_by_date = input_data.groupby(model.dateCol).apply(CalculateError)
            model.mean_error = errors_by_date.mean()
            std_dev_error = errors_by_date.std()
            model.u3sdl = recalCL
            model.u2sdl = warningCL
            model.l3sdl = -recalCL
            model.l2sdl = -warningCL
    
        return model.u2sdl, model.u3sdl, model.l2sdl, model.l3sdl
    
def SPCTrigger(model, input_data, dateCol='date', clStartDate=None, clEndDate=None, 
            numMonths=None, warningCL=None, recalCL=None, warningSDs=2, recalSDs=3, 
            verbose=True):
    """Trigger function to update the model if the error enters an upper control limit.
        The control limits can be set using one of the following methods:

        - Enter a start (clStartDate) and end date (clEndDate) to determine the control limits using the error mean and std during this period.
        - Enter the number of months (numMonths) to base the control limits on from the start of the period.
        - Manually set the control limits by entering the float values for the 'warning' (warningCL) and 'recalibration' (recalCL) zones.
        - Enter the number of standard deviations from the mean for the start of the warning zone (warningSDs) and the start of the recalibration zone (recalSDs).

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


    u2sdl, u3sdl, l2sdl, l3sdl = __SPCCalculateControlLimits(model, input_data, startCLDate, endCLDate, warningCL, recalCL, warningSDs, recalSDs)
    
    if verbose:
        print(f"SPC limits set:")
        print(f" - Upper 2-sigma limit: {u2sdl}")
        print(f" - Upper 3-sigma limit: {u3sdl}")
        print(f" - Lower 2-sigma limit: {l2sdl}")
        print(f" - Lower 3-sigma limit: {l3sdl}")

    return MethodType(lambda self, x: __SPCTrigger(self, x, model, u2sdl, u3sdl, l2sdl, l3sdl, endCLDate, verbose), model)

def __SPCTrigger(self, input_data, model, u2sdl, u3sdl, l2sdl, l3sdl, endCLDate, verbose):
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
    # if current date <= endCLDate then don't do anything (using that to set the limits)
    curDate = input_data[self.dateCol].max()
    if curDate <= endCLDate:
        return False
    
    _, error = Metrics.__SumOfDiffComputation(model, input_data, self.outcomeColName)
    
    # if error enter yellow zone (usually between 2SD and 3SD unless user has manually changed control limits) then print warning message
    if verbose:
        print(f'Mean error is currently {error}.\n')
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
    
