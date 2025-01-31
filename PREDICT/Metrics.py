import numpy as np

def Accuracy(model, outcomeCol='outcome', threshold=0.5):
    """
    LogHook to compute the accuracy of a model at each timestep.

    Args:
        model (PREDICTModel): The model to evaluate, must have a predict method.
        outcomeCol (str, optional): The column in the dataframe containing the actual outcomes. Defaults to 'outcome'.
        threshold (float, optional): Probability threshold at which to classify individuals. Defaults to 0.5.

    Returns:
        logHook: A hook to compute the accuracy of the model at each timestep when fed data.
    """
    return lambda df: __AccuracyComputation(model, df, outcomeCol, threshold)
    
def __AccuracyComputation(model, df, outcomeCol, threshold):
    """_summary_

    Args:
        model (PREDICTModel): The model to evaluate, must have a predict method.
        df (pd.DataFrame): DataFrame to evaluate the model on.
        outcomeCol (str, optional): The column in the dataframe containing the actual outcomes. Defaults to 'outcome'.
        threshold (float, optional): Probability threshold at which to classify individuals. Defaults to 0.5.

    Returns:
        hookname (str), result (float): The name of the hook ('Accuracy'), and the resulting accuracy of the model.
    """
    predictions = model.predict(df)
    classes = (predictions >= threshold).astype(int)
    outcomes = df[outcomeCol].astype(int)
    return 'Accuracy', np.mean(classes == outcomes)
