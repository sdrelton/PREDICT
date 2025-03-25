from sklearn.linear_model import LogisticRegression
import numpy as np
from scipy.special import logit



class PREDICTModel:
    """
    A class used to represent the PREDICT Model.
    """

    def __init__(self):
        """
        Initializes the model with empty lists for pre and post prediction hooks.
        """
        self.prePredictHooks = list()
        self.postPredictHooks = list()

    def predict(self, input_data):
        """
        Makes predictions based on the input data.

        Args:
            input_data (any): The input data for making predictions.
        """
        pass

    def trigger(self, input_data):
        """
        Evaluates whether the model needs to be updated based on the input data.

        Args: 
            input_data (any): The input data to evaluate the model update.

        Returns:
            bool: Returns False indicating no update is required.
        """
        return False

    def update(self, input_data):
        """
        Updates the model if required based on the input data.

        Args:
            input_data (any): The input data for updating the model.
        """
        pass

    def addPrePredictHook(self, hook):
        """
        Adds a hook to be executed before making predictions.

        Args:
            hook (callable): A function to be executed before predictions.
        """
        self.prePredictHooks.append(hook)
        
    def addPostPredictHook(self, hook):
        """
        Adds a hook to be executed after making predictions.

        Args:
            hook (callable): A function to be executed after predictions.
        """
        self.postPredictHooks.append(hook)
        


class EvaluatePredictions(PREDICTModel):
    """
    A class used to evaluate the predictions arising from another model which are already in the dataframe.
    
    Args:
        colName (str): The name of the column in the dataframe containing the predictions (default='prediction').
    """
    def __init__(self, colName='prediction'):
        self.colName = colName
        
    def predict(self, input_data):
        return input_data[self.colName]
    
    
class RecalibratePredictions(PREDICTModel):
    """
    A class which recalibrates the predictions arising from another model based on the trigger function.
    
    Recalibration involves using a logistic regression to adjust the model predictions.
    
    Needs to be followed by setting a trigger function (see example).
    
    Args:
        predictColName (str): The name of the column in the dataframe containing the predictions (default='prediction').
        outcomeColName (str): The name of the column in the dataframe containing the outcomes (default='outcome').
        dateCol (str): The name of the column in the dataframe containing the dates (default='date').
        
    Examples
    --------
    # Create a model which recalibrates predictions when triggered
    # Full example can be found in Examples/recalibration_example.ipynb
    model = RecalibratePredictions()
    model.trigger = AccuracyThreshold(model=model, threshold=0.7)       
    """
    
    def __init__(self, predictColName='prediction', outcomeColName='outcome', dateCol='date'):
        super(RecalibratePredictions, self).__init__()
        self.predictColName = predictColName
        self.outcomeColName = outcomeColName
        self.dateCol = dateCol
        
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __inverseSigmoid(self, y):
        return np.log(y / (1 - y))
        
    def predict(self, input_data):
        preds = input_data[self.predictColName]
        # Recalibrate from any hooks that have been added
        for hook in self.postPredictHooks:
            preds = hook(preds)
        return preds
    
    def update(self, input_data):
        # Get predictions
        preds = self.predict(input_data)
        
        # Convert to linear predictor scale
        lp = self.__inverseSigmoid(preds)
        
        # Work out model calibration
        logreg = LogisticRegression(penalty=None, max_iter=1000) # 'l1', 'elasticnet', 'l2' or None
        logreg.fit(np.array(lp).reshape(-1, 1), input_data[self.outcomeColName].astype(int))
        intercept = logreg.intercept_
        scale = logreg.coef_[0]
        
        # Add hook to adjust predictions accordingly
        recal = lambda p: self.__sigmoid(self.__inverseSigmoid(p) * scale + intercept)
        self.addPostPredictHook(recal)


    def CalculateControlLimits(self, input_data, startCLDate, endCLDate, warningCL, recalCL, warningSDs, recalSDs):
        """Calculate the static control limits of data using either the specific period (startCLDate to endCLDate), 
        the inputted control limits (warningCL, recalCL), or the first X months (nummMonths) since the start of the 
        data.

        Args:
            input_data (pd.DataFrame): The input data for updating the model.
            startCLDate (str): Start date to determine control limits from. Defaults to None.
            endCLDate (str): End date to determine control limits from. Defaults to None.
            warningCL (float): A manually set control limit for the warning control limit.
            recalCL (float): A manually set control limit for the recalibration trigger limit.
            warningSDs (int or float): Number of standard deviations from the mean to set the warning limit to. Defaults to 2.
            recalSDs (int or float): Number of standard deviations from the mean to set the recalibration trigger to. Defaults to 3.

        Returns:
            float, float: Two upper control limits for the warning and danger/recalibration trigger zones.
        """
        def CalculateError(group):
            predictions = group['predictions']
            differences = group[self.outcomeColName] - predictions
            sum_of_differences = np.sum(differences)/len(group[self.outcomeColName])
            return sum_of_differences
        
        if startCLDate is not None and endCLDate is not None:
            # Get the logreg error at each timestep within the control limit determination period
            createCLdf = input_data[(input_data[self.dateCol] >= startCLDate) & (input_data[self.dateCol] <= endCLDate)].copy()
            
            # Predictions column
            createCLdf['predictions'] = self.predict(createCLdf)

            errors_by_date = createCLdf.groupby(self.dateCol).apply(CalculateError)
            self.mean_error = errors_by_date.mean()
            std_dev_error = errors_by_date.std()
            
            self.u2sdl = self.mean_error + warningSDs * std_dev_error
            self.u3sdl = self.mean_error + recalSDs * std_dev_error
            self.l2sdl = self.mean_error - warningSDs * std_dev_error
            self.l3sdl = self.mean_error - recalSDs * std_dev_error

        #elif warningCL is not None and recalCL is not None:
        else:
            # Predictions column
            input_data['predictions'] = self.predict(input_data)
            errors_by_date = input_data.groupby(self.dateCol).apply(CalculateError)
            self.mean_error = errors_by_date.mean()
            std_dev_error = errors_by_date.std()
            self.u3sdl = recalCL
            self.u2sdl = warningCL
            self.l3sdl = -recalCL
            self.l2sdl = -warningCL
    
        return self.u2sdl, self.u3sdl, self.l2sdl, self.l3sdl
