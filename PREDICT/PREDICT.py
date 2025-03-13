import pandas as pd
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
        or an integer representing the number of days.
    currentWindowStart : pd.datetime
        The current start date of the prediction window.
    currentWindowEnd : pd.datetime
        The current end date of the prediction window.
    log : dict
        A dictionary to store logs.
    logHooks : list
        A list of hooks to be called during logging.
    """
    
    def __init__(self, data, model, dateCol = 'date', startDate='min', endDate='max', timestep='week'):
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

        if timestep == 'week':
            self.timestep = pd.Timedelta(weeks=1)
        elif timestep == 'day':
            self.timestep = pd.Timedelta(days=1)
        elif timestep == 'month':
            self.timestep = pd.Timedelta(weeks=4)
        else:
            self.timestep = pd.Timedelta(weeks=timestep)

        self.currentWindowStart = self.startDate
        self.currentWindowEnd = self.startDate + self.timestep
        self.log = dict()
        self.logHooks = list()

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
        while self.currentWindowEnd <= self.endDate:
            self.model.predict(self.data)
            dates = self.data[self.dateCol]
            curdata = self.data[(dates >= self.currentWindowStart) & (dates < self.currentWindowEnd)]
            for hook in self.logHooks:
                logname, result = hook(curdata)
                self.addLog(logname, self.currentWindowEnd, result)
            if self.model.trigger(curdata):
                self.model.update(curdata)
                # Add to log
                self.addLog('Model Updated', self.currentWindowEnd, True)
            self.currentWindowStart += self.timestep
            self.currentWindowEnd += self.timestep