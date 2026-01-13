import pandas as pd
import numpy as np
import datetime as dt
from PREDICT.Models import RecalibratePredictions


class DummyLogReg:
    def __init__(self, penalty=None, max_iter=None):
        self.intercept_ = None
        self.coef_ = None
        self.fitted_X = None
        self.fitted_y = None

    def fit(self, X, y):
        # Record inputs for assertions and set deterministic coef/intercept
        self.fitted_X = np.array(X).reshape(-1).tolist()
        self.fitted_y = np.array(y).reshape(-1).tolist()
        # deterministic values for tests depending on input length
        n = len(self.fitted_X)
        if n >= 10:
            self.intercept_ = np.array([0.1])
            self.coef_ = np.array([0.5])
        elif n <= 3:
            self.intercept_ = np.array([0.2])
            self.coef_ = np.array([0.6])
        else:
            self.intercept_ = np.array([0.15])
            self.coef_ = np.array([0.55])
        return self


def make_dataframe(start_date, n_days):
    dates = [start_date + dt.timedelta(days=i) for i in range(n_days)]
    # simple predictions between 0.2 and 0.8, outcomes alternating 0/1
    preds = [0.2 + 0.6 * (i % 2) for i in range(n_days)]
    outcomes = [i % 2 for i in range(n_days)]
    return pd.DataFrame({
        'date': pd.to_datetime(dates),
        'prediction': pd.Series(preds),
        'outcome': pd.Series(outcomes)
    })


def test_recalibrate_update_uses_full_window_when_recalperiod_none(monkeypatch):
    start = dt.datetime(2020, 1, 1)
    df = make_dataframe(start, 10)

    # Patch the LogisticRegression used inside RecalibratePredictions.update
    monkeypatch.setattr('PREDICT.Models.LogisticRegression', DummyLogReg)

    model = RecalibratePredictions(predictColName='prediction', outcomeColName='outcome', dateCol='date')

    intercept, scale = model.update(df, windowStart=pd.Timestamp(start), windowEnd=pd.Timestamp(start + dt.timedelta(days=10)), recalPeriod=None)

    # After update, there should be one postPredictHook added
    assert len(model.postPredictHooks) == 1
    # The returned intercept and scale are from our DummyLogReg
    assert np.isclose(intercept, np.array([0.1])).all()
    assert np.isclose(scale, np.array([0.5])).all()


def test_recalibrate_update_respects_recalperiod(monkeypatch):
    start = dt.datetime(2020, 1, 1)
    df = make_dataframe(start, 10)

    # Patch the LogisticRegression used inside RecalibratePredictions.update
    monkeypatch.setattr('PREDICT.Models.LogisticRegression', DummyLogReg)

    model = RecalibratePredictions(predictColName='prediction', outcomeColName='outcome', dateCol='date')

    # Use a recalPeriod of 3 days: should use only last 3 rows (dates 2020-01-08,09,10)
    intercept, scale = model.update(df, windowStart=pd.Timestamp(start), windowEnd=pd.Timestamp(start + dt.timedelta(days=10)), recalPeriod=3)

    assert len(model.postPredictHooks) == 1
    # For recalPeriod=3 we expect the dummy regressor to have seen 3 samples
    assert np.isclose(intercept, np.array([0.2])).all()
    assert np.isclose(scale, np.array([0.6])).all()
