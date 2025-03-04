import numpy as np
import sklearn as skl

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
    """
    Function to compute the accuracy of a model on a given dataframe.

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

def AUROC(model, outcomeCol='outcome'):
    """
    LogHook to compute the AUROC of a model at each timestep.

    Args:
        model (PREDICTModel): The model to evaluate, must have a predict method.
        outcomeCol (str, optional): The column in the dataframe containing the actual outcomes. Defaults to 'outcome'.

    Returns:
        logHook: A hook to compute the AUROC of the model at each timestep when fed data.
    """
    return lambda df: __AUROCComputation(model, df, outcomeCol)

def __AUROCComputation(model, df, outcomeCol):
    """
    Function to compute the AUROC of a model on a given dataframe.

    Args:
        model (PREDICTModel): The model to evaluate, must have a predict method.
        df (pd.DataFrame): DataFrame to evaluate the model on.
        outcomeCol (str, optional): The column in the dataframe containing the actual outcomes. Defaults to 'outcome'.

    Returns:
        hookname (str), result (float): The name of the hook ('AUROC'), and the resulting AUROC of the model.
    """
    predictions = model.predict(df)
    fpr, tpr, _ = skl.metrics.roc_curve(df[outcomeCol], predictions)
    return 'AUROC', skl.metrics.auc(fpr, tpr)

def F1Score(model, outcomeCol='outcome', threshold=0.5):
    """
    LogHook to compute the F1 score of a model at each timestep.

    Args:
        model (PREDICTModel): The model to evaluate, must have a predict method.
        outcomeCol (str, optional): The column in the dataframe containing the actual outcomes. Defaults to 'outcome'.
        threshold (float, optional): Probability threshold at which to classify individuals. Defaults to 0.5.

    Returns:
        logHook: A hook to compute the F1 score of the model at each timestep when fed data.
    """
    return lambda df: __F1scoreComputation(model, df, outcomeCol, threshold)

def __F1scoreComputation(model, df, outcomeCol, threshold):
    """
    Function to compute the F1 score of a model on a given dataframe.

    Args:
        model (PREDICTModel): The model to evaluate, must have a predict method.
        df (pd.DataFrame): DataFrame to evaluate the model on.
        outcomeCol (str, optional): The column in the dataframe containing the actual outcomes. Defaults to 'outcome'.

    Returns:
        hookname (str), result (float): The name of the hook ('F1score'), and the resulting F1 score of the model.
    """
    predictions = model.predict(df)
    classes = (predictions >= 0.5).astype(int)
    outcomes = df[outcomeCol].astype(int)
    tp = np.sum((classes == 1) & (outcomes == 1))
    fp = np.sum((classes == 1) & (outcomes == 0))
    fn = np.sum((classes == 0) & (outcomes == 1))
    return 'F1score', 2 * tp / (2 * tp + fp + fn)

def Precision(model, outcomeCol='outcome', threshold=0.5):
    """
    LogHook to compute the precision of a model at each timestep.

    Args:
        model (PREDICTModel): The model to evaluate, must have a predict method.
        outcomeCol (str, optional): The column in the dataframe containing the actual outcomes. Defaults to 'outcome'.
        threshold (float, optional): Probability threshold at which to classify individuals. Defaults to 0.5.

    Returns:
        logHook: A hook to compute the precision of the model at each timestep when fed data.
    """
    return lambda df: __PrecisionComputation(model, df, outcomeCol, threshold)

def __PrecisionComputation(model, df, outcomeCol, threshold):
    """
    Function to compute the precision of a model on a given dataframe.

    Args:
        model (PREDICTModel): The model to evaluate, must have a predict method.
        df (pd.DataFrame): DataFrame to evaluate the model on.
        outcomeCol (str, optional): The column in the dataframe containing the actual outcomes. Defaults to 'outcome'.

    Returns:
        hookname (str), result (float): The name of the hook ('Precision'), and the resulting precision of the model.
    """
    predictions = model.predict(df)
    classes = (predictions >= 0.5).astype(int)
    outcomes = df[outcomeCol].astype(int)
    tp = np.sum((classes == 1) & (outcomes == 1))
    fp = np.sum((classes == 1) & (outcomes == 0))
    return 'Precision', tp / (tp + fp)

def Recall(model, outcomeCol='outcome', threshold=0.5):
    """
    LogHook to compute the recall of a model at each timestep.

    Args:
        model (PREDICTModel): The model to evaluate, must have a predict method.
        outcomeCol (str, optional): The column in the dataframe containing the actual outcomes. Defaults to 'outcome'.
        threshold (float, optional): Probability threshold at which to classify individuals. Defaults to 0.5.

    Returns:
        logHook: A hook to compute the recall of the model at each timestep when fed data.
    """
    return lambda df: __RecallComputation(model, df, outcomeCol, threshold)

def __RecallComputation(model, df, outcomeCol, threshold):
    """
    Function to compute the recall of a model on a given dataframe.

    Args:
        model (PREDICTModel): The model to evaluate, must have a predict method.
        df (pd.DataFrame): DataFrame to evaluate the model on.
        outcomeCol (str, optional): The column in the dataframe containing the actual outcomes. Defaults to 'outcome'.

    Returns:
        hookname (str), result (float): The name of the hook ('Recall'), and the resulting recall of the model.
    """
    predictions = model.predict(df)
    classes = (predictions >= 0.5).astype(int)
    outcomes = df[outcomeCol].astype(int)
    tp = np.sum((classes == 1) & (outcomes == 1))
    fn = np.sum((classes == 0) & (outcomes == 1))
    return 'Recall', tp / (tp + fn)

def Sensitivity(model, outcomeCol='outcome', threshold=0.5):
    """
    LogHook to compute the sensitivity of a model at each timestep.

    Args:
        model (PREDICTModel): The model to evaluate, must have a predict method.
        outcomeCol (str, optional): The column in the dataframe containing the actual outcomes. Defaults to 'outcome'.
        threshold (float, optional): Probability threshold at which to classify individuals. Defaults to 0.5.

    Returns:
        logHook: A hook to compute the sensitivity of the model at each timestep when fed data.
    """
    return lambda df: __SensitivityComputation(model, df, outcomeCol, threshold)

def __SensitivityComputation(model, df, outcomeCol, threshold):
    """
    Function to compute the sensitivity of a model on a given dataframe.

    Args:
        model (PREDICTModel): The model to evaluate, must have a predict method.
        df (pd.DataFrame): DataFrame to evaluate the model on.
        outcomeCol (str, optional): The column in the dataframe containing the actual outcomes. Defaults to 'outcome'.

    Returns:
        hookname (str), result (float): The name of the hook ('Sensitivity'), and the resulting sensitivity of the model.
    """
    predictions = model.predict(df)
    classes = (predictions >= 0.5).astype(int)
    outcomes = df[outcomeCol].astype(int)
    tp = np.sum((classes == 1) & (outcomes == 1))
    fn = np.sum((classes == 0) & (outcomes == 1))
    return 'Sensitivity', tp / (tp + fn)

def Specificity(model, outcomeCol='outcome', threshold=0.5):
    """
    LogHook to compute the specificity of a model at each timestep.

    Args:
        model (PREDICTModel): The model to evaluate, must have a predict method.
        outcomeCol (str, optional): The column in the dataframe containing the actual outcomes. Defaults to 'outcome'.
        threshold (float, optional): Probability threshold at which to classify individuals. Defaults to 0.5.

    Returns:
        logHook: A hook to compute the specificity of the model at each timestep when fed data.
    """
    return lambda df: __SpecificityComputation(model, df, outcomeCol, threshold)

def __SpecificityComputation(model, df, outcomeCol, threshold):
    """
    Function to compute the specificity of a model on a given dataframe.

    Args:
        model (PREDICTModel): The model to evaluate, must have a predict method.
        df (pd.DataFrame): DataFrame to evaluate the model on.
        outcomeCol (str, optional): The column in the dataframe containing the actual outcomes. Defaults to 'outcome'.

    Returns:
        hookname (str), result (float): The name of the hook ('Specificity'), and the resulting specificity of the model.
    """
    predictions = model.predict(df)
    classes = (predictions >= 0.5).astype(int)
    outcomes = df[outcomeCol].astype(int)
    tn = np.sum((classes == 0) & (outcomes == 0))
    fp = np.sum((classes == 1) & (outcomes == 0))
    return 'Specificity', tn / (tn + fp)

