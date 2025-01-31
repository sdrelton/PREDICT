import pandas as pd
class PREDICT:
    """
    A class used to represent the PREDICT model.
    Attributes
    ----------
    data : any
        The data to be used by the model.
    model : any
        The model to be used for prediction.
    dateCol: str
        The column in the data that contains the date.
    startDate : any
        The start date for the prediction window.
    endDate : any
        The end date for the prediction window.
    timestep : any
        The timestep for the prediction window.
    currentWindowStart : any
        The current start date of the prediction window.
    currentWindowEnd : any
        The current end date of the prediction window.
    log : dict
        A dictionary to store logs.
    logHooks : list
        A list of hooks to be called during logging.
    Methods
    -------
    addLogHook(hook)
        Adds a hook to the logHooks list.
    addLog(key, date, val)
        Adds a log entry to the log dictionary.
    getLog()
        Returns the log dictionary.
    run()
        Runs the prediction model over the specified date range.
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
        else:
            self.timestep = timestep
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
            if self.model.trigger(self.data):
                self.model.update(self.data)
                # Add to log
            self.currentWindowStart += self.timestep
            self.currentWindowEnd += self.timestep