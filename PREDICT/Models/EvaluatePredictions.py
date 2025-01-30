from PREDICT.Models.PREDICTModel import PREDICTModel

class EvaluatePredictions(PREDICTModel):
    """
    A class used to evaluate the predictions arising from another model which are already in the dataframe.
    
    Fields
    ------
    colName : str
        The name of the column in the dataframe containing the predictions (default='prediction').
    """
    def __init__(self, colName='prediction'):
        self.colName = colName
        
    def predict(self, input_data):
        return input_data[self.colName]