import numpy as np

def Accuracy(model, outcomeCol='outcome', threshold=0.5):
    return lambda df: AccuracyComputation(model, df, outcomeCol, threshold)
    
def AccuracyComputation(model, df, outcomeCol, threshold):
    predictions = model.predict(df)
    classes = (predictions > threshold).astype(int)
    outcomes = df[outcomeCol].astype(int)
    return 'Accuracy', np.mean(classes == outcomes)