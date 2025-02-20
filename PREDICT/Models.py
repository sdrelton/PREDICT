from sklearn.linear_model import LogisticRegression
import numpy as np

def __sigmoid(x):
    return 1 / (1 + np.exp(-x))

def __inverse_sigmoid(y):
    return np.log(y / (1 - y))

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

        Parameters
        ----------
        input_data : any
            The input data for making predictions.
        """
        pass

    def trigger(self, input_data):
        """
        Evaluates whether the model needs to be updated based on the input data.

        Parameters
        ----------
        input_data : any
            The input data to evaluate the model update.

        Returns
        -------
        bool
            Returns False indicating no update is required.
        """
        return False

    def update(self, input_data):
        """
        Updates the model if required based on the input data.

        Parameters
        ----------
        input_data : any
            The input data for updating the model.
        """
        pass

    def addPrePredictHook(self, hook):
        """
        Adds a hook to be executed before making predictions.

        Parameters
        ----------
        hook : callable
            A function to be executed before predictions.
        """
        self.prePredictHooks.append(hook)
        
    def addPostPredictHook(self, hook):
        """
        Adds a hook to be executed after making predictions.

        Parameters
        ----------
        hook : callable
            A function to be executed after predictions.
        """
        self.postPredictHooks.append(hook)
        


class EvaluatePredictions(PREDICTModel):
    """
    A class used to evaluate the predictions arising from another model which are already in the dataframe.
    
    Attributes
    ------
    colName : str
        The name of the column in the dataframe containing the predictions (default='prediction').
    """
    def __init__(self, colName='prediction'):
        self.colName = colName
        
    def predict(self, input_data):
        return input_data[self.colName]
    
    
class RecalibratePredictions(PREDICTModel):
    """
    A class which recalibrates the predictions arising from another model based on the trigger function.
    
    Recalibration involves using a logistic regression to adjust the model predictions.
    
    Attributes
    ----------
    colName: str
        The name of the column in the dataframe containing the predictions (default='prediction').
        
    trigger: callable
        A function which determines whether the model needs to be updated.
    """
    
    def __init__(self, predictColName='prediction', outcomeColName='outcome', trigger=None):
        super(RecalibratePredictions, self).__init__()
        self.predictColName = predictColName
        self.outcomeColName = outcomeColName
        self.trigger = trigger
        
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
        lp = __inverse_sigmoid(preds)
        
        # Work out model calibration
        logreg = LogisticRegression(penalty=None, max_iter=1000)
        logreg.fit(lp.reshape(-1, 1), input_data[self.outcomeColName])
        intercept = logreg.intercept_
        scale = logreg.coef_[0]
        
        # Add hook to adjust predictions accordingly
        recal = lambda p: __sigmoid(__inverse_sigmoid(p) * scale + intercept)
        self.addPostPredictHook(recal)