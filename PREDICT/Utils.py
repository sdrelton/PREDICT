import numpy as np
from scipy.stats import entropy
import pandas as pd

def bayesian_no_constants(df):
    """
    Ensures no constant values within the dataframe, which would cause Bambi to crash.
    If there is such a column, flip a random value.

    Args:
        df (pd.DataFrame): The input dataframe to check for constant values.
    """
    for col in df.columns:
        if df[col].nunique() == 1:
            try:
                # Assume column is binary and flip one of the row values
                nrows = len(df[col])
                ix = np.random.choice(nrows)
                df[col].iloc[ix] = 1 - df[col].iloc[ix]
            except Exception as e:
                print(f'Could not process constants for column {col}')
    return df

def kl_divergence(x, y, n_bins=100, epsilon=1e-10):
    """
    Calculates the KL divergence between two continuous distributions, approximated by binning.

    Args:
        x (np.ndarray): Samples from the first distribution (e.g., initial residuals).
        y (np.ndarray): Samples from the second distribution (e.g., new residuals).
        n_bins (int): Number of bins to use for the histogram.
        epsilon (float): Small value to avoid division by zero.

    Returns:
        float: The KL divergence.
    """
    # Define the range of the histograms based on the combined data
    min_val = min(np.min(x), np.min(y))
    max_val = max(np.max(x), np.max(y))
    bins = np.linspace(min_val, max_val, n_bins + 1)

    # Create histograms for both sets of residuals
    p, _ = np.histogram(x, bins=bins, density=True)
    q, _ = np.histogram(y, bins=bins, density=True)

    # Add a small epsilon to avoid zeros
    p += epsilon
    q += epsilon

    # Normalize to get probability distributions
    p /= np.sum(p)
    q /= np.sum(q)

    return entropy(p, q)

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