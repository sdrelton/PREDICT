import numpy as np

def accuracy_threshold(threshold, prediction_threshold=0.5):
    return lambda x: _accuracy_threshold(x, threshold, prediction_threshold)

def _accuracy_threshold(self, input_data, threshold, prediction_threshold):
    """
    Trigger function to update model if accuracy falls below a given threshold.
    
    TODO: Fill in
    """
    preds = self.predict(input_data)
    outcomes = input_data[self.outcomeColName].astype(int)
    preds_rounded = np.array(preds >= prediction_threshold).astype(int)
    accuracy = np.mean(preds_rounded == outcomes)
    if accuracy >= threshold:
        return False
    else:
        return True
    
    