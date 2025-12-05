import pandas as pd
from dateutil.relativedelta import relativedelta
import logging

class PREDICT:
    """
    A class used to represent the PREDICT model.
    
    Attributes
    ----------
    data : pd.DataFrame
        The data to be used by the model.
    model : PREDICTModel
        The model to be used for prediction.
    dateCol: str
        The column in the data that contains the date.
    startDate : str, dt.datetime
        The start date for the prediction window.
    endDate : str, pd.datetime
        The end date for the prediction window.
    timestep : str, int
        The timestep for the prediction window. Must be 'week', 'day', 'month', 
        or an integer representing the number of days. Defaults to 'month'.
    currentWindowStart : pd.datetime
        The current start date of the prediction window.
    currentWindowEnd : pd.datetime
        The current end date of the prediction window.
    log : dict
        A dictionary to store logs.
    logHooks : list
        A list of hooks to be called during logging.
    recal_period : int
        An integer giving the number of days for the recalibration window. 
        Defaults to one year (365).
    saveRecalibratedPredictions : bool
        Save recalibrated predictions into a csv file. Defaults to False.
    model_name : str
        Name of the model, used to name csv with recalibrated predictions.
    verbose : bool
        Print sample size calculation warnings.
    """
    
    def __init__(self, data, model, model_name, dateCol = 'date', startDate='min', endDate='max', timestep='month', recal_period=365, verbose=False, startOfAnalysis=None):
        """
        Initializes the PREDICT class with default values.
        """
        self.data = data
        self.model = model
        self.dateCol = dateCol
        if startDate == 'min':
            self.startDate = self.data[self.dateCol].min()
        else:
            self.startDate = startDate

        if endDate == 'max':
            self.endDate = self.data[self.dateCol].max()
        else:
            self.endDate = endDate
        try:
            if timestep == 'week':
                self.timestep = pd.Timedelta(weeks=1)
            elif timestep == 'day':
                self.timestep = pd.Timedelta(days=1)
            elif timestep == 'month':
                self.timestep = relativedelta(months=1)
            elif isinstance(timestep, int):
                self.timestep = pd.Timedelta(weeks=timestep)
            else:
                raise TypeError
        except (ValueError, TypeError):
            print("Invalid timestep value, timestep must be 'week', 'day', 'month' or an integer representing days. Defaulting to 'week'.")
            self.timestep = pd.Timedelta(weeks=1)

        self.currentWindowStart = self.startDate
        self.currentWindowEnd = self.startDate + self.timestep
        self.log = dict()
        self.logHooks = list()
        self.verbose = verbose
        self.intercept = 0
        self.scale = 0
        self.recal_period = recal_period
        self.model_name = model_name
        self.startOfAnalysis = startOfAnalysis

    def addLogHook(self, hook):
        """
        Adds a hook to the logHooks list.
        Parameters
        ----------
        hook : function
            A function to be called during logging.
        """
        self.logHooks.append(hook)
    
    def addLog(self, key, date, val):
        """
        Adds a log entry to the log dictionary.
        Parameters
        ----------
        key : str
            The key for the log entry.
        date : any
            The date for the log entry.
        val : any
            The value for the log entry.
        """
        if key not in self.log.keys():
            self.log[key] = dict()
        self.log[key][date] = val
    
    def getLog(self):
        """
        Returns the log dictionary.
        Returns
        -------
        dict
            The log dictionary.
        """
        return self.log
    
    def run(self):
        """
        Runs the prediction model over the specified date range.
        """
        n_samples = 0
        while self.currentWindowEnd <= self.endDate:
            dates = self.data[self.dateCol]
            curdata = self.data[(dates >= self.currentWindowStart) & (dates < self.currentWindowEnd)]
            for hook in self.logHooks:
                logname, result = hook(curdata)
                self.addLog(logname, self.currentWindowEnd, result)
            # append the updated prediction to the prediction and outcomes file
            # get the current predictions for the window in certain conditions to prevent multiple writes caused by overlap
            if 'bayesian' in self.model_name.lower() and ((self.currentWindowStart==self.startOfAnalysis) or (self.currentWindowStart > self.startDate)):
                print("Saving predictors and outcomes...")
                
                # save them to the predictions and outcomes csv file alongside the dates and outcomes
                predsdf = pd.DataFrame({'date': list(curdata[self.dateCol]), 'outcome': list(curdata['outcome']), 'prediction': self.model.predict(curdata)})
                write_mode = 'w' if self.currentWindowStart==self.startOfAnalysis else 'a'
                header_type = True if self.currentWindowStart==self.startOfAnalysis else False
                predsdf.to_csv(f"{self.model_name}_predictions_and_outcomes.csv", mode=write_mode, header=header_type, index=False)

            if self.model.trigger(curdata):
                print("Trigger activated")
                # update based on the chosen window of data
                if self.recal_period == 30: # if 30 days aka a month is inputted then use the currentWindowStart instead
                    update_data = self.data[(dates >= (self.currentWindowEnd - relativedelta(months=1))) & (dates < self.currentWindowEnd)]
                else:
                    update_data = self.data[(dates >= (self.currentWindowEnd - pd.Timedelta(days=self.recal_period))) & (dates < self.currentWindowEnd)]
                if 'bayesian' in self.model_name.lower():
                    self.model.update(update_data)
                    
                else:
                    self.intercept, self.scale = self.model.update(update_data)
                    print(f"Intercept: {self.intercept}\nScale: {self.scale}")
                # Add to log
                self.addLog('Model Updated', self.currentWindowEnd, True)
                # if verbose and trigger happens then do sample size calculation for the window
                if self.verbose:
                    n_samples = len(update_data)
                    # Sample size calculation
                    n_features = self.data.shape[1] - 3 # minus date, prediction and label columns
                    if n_samples < 10 * n_features: # sample size should be at least 10 times the number of features
                        logging.warning(f"Warning: Sample size ({n_samples}) is less than 10 times the number of features ({n_features}). Model performance may be unreliable.")        

            self.currentWindowStart += self.timestep
            self.currentWindowEnd += self.timestep