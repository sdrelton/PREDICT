import pandas as pd
import numpy as np
from PREDICT.Utils import StorePredictions


class DummyModel:
    def __init__(self, preds):
        self._preds = pd.Series(preds)

    def predict(self, df):
        # Return a pandas Series to match typical behaviour
        return self._preds


def test_store_predictions_hook_returns_name_and_predictions():
    model = DummyModel([0.1, 0.9])
    df = pd.DataFrame({"p": [1, 2]})

    hook = StorePredictions(model)
    name, preds = hook(df)

    assert name == 'StorePredictions'
    # Ensure the predictions returned are exactly those from the model
    assert list(preds) == [0.1, 0.9]
