import numpy as np

def StorePredictions(model):
    """
    Loghook to store model predictions.
    Args:
        model (PREDICTModel): The model to evaluate, must have a predict method.
    
    Returns:
        logHook: A loghook function that stores predictions in the log.
    """
    
    def __predictHook(model, df):
        """
        Function to compute the current model prediction on window of data

        Args:
            model (PREDICTModel): The model to evaluate, must have a predict method.
            df (pd.DataFrame): DataFrame to evaluate the model on.
            outcomeCol (str, optional): The column in the dataframe containing the actual outcomes. Defaults to 'outcome'.
        
        Returns:
            hookname (str), result (float): The name of the hook ('StorePredictions'), and the resulting predictions of the model.
        """
        return 'StorePredictions', model.predict(df)    
    
    return lambda x: __predictHook(model, x)