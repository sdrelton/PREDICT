from PREDICT import Metrics
import numpy as np
import pandas as pd


class MockModel:
    def predict(self, df):
        return df['probability']

def test_accuracy_computation():
    model = MockModel()
    data = {
        'probability': [0.1, 0.4, 0.6, 0.9],
        'outcome': [0, 0, 1, 1]
    }
    df = pd.DataFrame(data)
    
    accuracy = Metrics.Accuracy(model, 'outcome', 0.5)
    hookname, result = accuracy(df)

    assert hookname == 'Accuracy'
    assert result == 1.0