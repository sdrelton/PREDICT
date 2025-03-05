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

def AUPRC(model, outcomeCol='outcome'):
    """
    LogHook to compute the AUPRC of a model at each timestep.

    Args:
        model (PREDICTModel): The model to evaluate, must have a predict method.
        outcomeCol (str, optional): The column in the dataframe containing the actual outcomes. Defaults to 'outcome'.

    Returns:
        logHook: A hook to compute the AUPRC of the model at each timestep when fed data.
    """
    return lambda df: __AUPRCComputation(model, df, outcomeCol)

def __AUPRCComputation(model, df, outcomeCol):
    """
    Function to compute the AUPRC of a model on a given dataframe.

    Args:
        model (PREDICTModel): The model to evaluate, must have a predict method.
        df (pd.DataFrame): DataFrame to evaluate the model on.
        outcomeCol (str, optional): The column in the dataframe containing the actual outcomes. Defaults to 'outcome'.

    Returns:
        hookname (str), result (float): The name of the hook ('AUPRC'), and the resulting AUPRC of the model.
    """
    predictions = model.predict(df)
    precision, recall, _ = skl.metrics.precision_recall_curve(df[outcomeCol], predictions)
    return 'AUPRC', skl.metrics.auc(recall, precision)


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
        threshold (float, optional): Probability threshold at which to classify individuals. Defaults to 0.5.

    Returns:
        hookname (str), result (float): The name of the hook ('F1score'), and the resulting F1 score of the model.
    """
    predictions = model.predict(df)
    classes = (predictions >= threshold).astype(int)
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
        threshold (float, optional): Probability threshold at which to classify individuals.

    Returns:
        hookname (str), result (float): The name of the hook ('Precision'), and the resulting precision of the model.
    """
    predictions = model.predict(df)
    classes = (predictions >= threshold).astype(int)
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
        threshold (float, optional): Probability threshold at which to classify individuals.

    Returns:
        hookname (str), result (float): The name of the hook ('Recall'), and the resulting recall of the model.
    """
    predictions = model.predict(df)
    classes = (predictions >= threshold).astype(int)
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
        threshold (float, optional): Probability threshold at which to classify individuals.

    Returns:
        hookname (str), result (float): The name of the hook ('Sensitivity'), and the resulting sensitivity of the model.
    """
    predictions = model.predict(df)
    classes = (predictions >= threshold).astype(int)
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
        threshold (float, optional): Probability threshold at which to classify individuals.

    Returns:
        hookname (str), result (float): The name of the hook ('Specificity'), and the resulting specificity of the model.
    """
    predictions = model.predict(df)
    classes = (predictions >= threshold).astype(int)
    outcomes = df[outcomeCol].astype(int)
    tn = np.sum((classes == 0) & (outcomes == 0))
    fp = np.sum((classes == 1) & (outcomes == 0))
    return 'Specificity', tn / (tn + fp)


def CalibrationSlope(model, outcomeCol='outcome'):
    """
    LogHook to compute the calibration slope of a model at each timestep.

    Args:
        model (PREDICTModel): The model to evaluate, must have a predict method.
        outcomeCol (str, optional): The column in the dataframe containing the actual outcomes. Defaults to 'outcome'.

    Returns:
        logHook: A hook to compute the calibration slope of the model at each timestep when fed data.
    """
    return lambda df: __CalibrationSlopeComputation(model, df, outcomeCol)

def __CalibrationSlopeComputation(model, df, outcomeCol):
    """
    Function to compute the calibration slope of a model on a given dataframe.

    Args:
        model (PREDICTModel): The model to evaluate, must have a predict method.
        df (pd.DataFrame): DataFrame to evaluate the model on.
        outcomeCol (str, optional): The column in the dataframe containing the actual outcomes. Defaults to 'outcome'.

    Returns:
        hookname (str), result (float): The name of the hook ('CalibrationSlope'), and the resulting calibration slope of the model.
    """
    predictions = model.predict(df)
    predictions_np = predictions.to_numpy().reshape(-1, 1)
    probas = skl.preprocessing.StandardScaler().fit_transform(predictions_np)
    slope = skl.linear_model.LinearRegression().fit(probas, df[outcomeCol].to_numpy()).coef_[0]
    return 'CalibrationSlope', slope



def CITL(model, outcomeCol='outcome'):
    """
    LogHook to compute the Calibration-In-The-Large (CITL) of a model at each timestep.

    Args:
        model (PREDICTModel): The model to evaluate, must have a predict method.
        outcomeCol (str, optional): The column in the dataframe containing the actual outcomes. Defaults to 'outcome'.

    Returns:
        logHook: A hook to compute the CITL of the model at each timestep when fed data.
    """
    return lambda df: __CITLComputation(model, df, outcomeCol)

def __CITLComputation(model, df, outcomeCol):
    """
    Function to compute the Calibration-In-The-Large (CITL) of a model on a given dataframe.

    Args:
        model (PREDICTModel): The model to evaluate, must have a predict method.
        df (pd.DataFrame): DataFrame to evaluate the model on.
        outcomeCol (str, optional): The column in the dataframe containing the actual outcomes. Defaults to 'outcome'.

    Returns:
        hookname (str), result (float): The name of the hook ('CITL'), and the resulting CITL of the model.
    """
    predictions = model.predict(df)
    mean_pred = predictions.mean()
    mean_outcome = df[outcomeCol].mean()
    citl = mean_pred - mean_outcome
    return 'CITL', citl

def OE(model, outcomeCol='outcome'):
    """
    LogHook to compute the Observed/Expected (O/E) ratio of a model at each timestep.

    Args:
        model (PREDICTModel): The model to evaluate, must have a predict method.
        outcomeCol (str, optional): The column in the dataframe containing the actual outcomes. Defaults to 'outcome'.

    Returns:
        logHook: A hook to compute the O/E ratio of the model at each timestep when fed data.
    """
    return lambda df: __OEComputation(model, df, outcomeCol)

def __OEComputation(model, df, outcomeCol):
    """
    Function to compute the Observed/Expected (O/E) ratio of a model on a given dataframe.

    Args:
        model (PREDICTModel): The model to evaluate, must have a predict method.
        df (pd.DataFrame): DataFrame to evaluate the model on.
        outcomeCol (str, optional): The column in the dataframe containing the actual outcomes. Defaults to 'outcome'.

    Returns:
        hookname (str), result (float): The name of the hook ('O/E'), and the resulting O/E ratio of the model.
    """
    predictions = model.predict(df)
    mean_pred = predictions.mean()
    mean_outcome = df[outcomeCol].mean()
    oe_ratio = mean_outcome / mean_pred if mean_pred != 0 else float('inf')
    return 'O/E', oe_ratio


