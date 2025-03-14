import numpy as np
from PREDICT import Models, Triggers, PREDICT
import pandas as pd
import datetime as dt
from PREDICT.Metrics import Accuracy

def test_add_hooks():
    """Test that hooks can be added to the model.
    """
    model = Models.PREDICTModel()
    pre_hook = lambda x: x
    post_hook = lambda x: x

    model.addPrePredictHook(pre_hook)
    model.addPostPredictHook(post_hook)

    assert len(model.prePredictHooks) == 1
    assert len(model.postPredictHooks) == 1
    assert model.prePredictHooks[0] == pre_hook
    assert model.postPredictHooks[0] == post_hook

def test_evaluate_predictions():
    """Test that the evaluate predictions model works.
    """
    data = {'prediction': np.array([1, 0, 1])}
    model = Models.EvaluatePredictions()
    predictions = model.predict(data)

    assert np.array_equal(predictions, data['prediction'])

def test_recalibrate_predictions():
    """Test that the recalibrate predictions model produces a non-None output
        and checking the recalibration hook is added.
    """
    data = {'prediction': np.array([0.2, 0.5, 0.8]), 'outcome': np.array([0, 1, 1])}
    model = Models.RecalibratePredictions()
    model.update(data)
    recalibrated_predictions = model.predict(data)

    assert recalibrated_predictions is not None
    assert len(recalibrated_predictions) == len(data['prediction'])
    assert model.postPredictHooks

def test_regular_recalibrations():
    """Test that the regular recalibrations model updates when expected.
    """

    mydict = {
        'date': list(),
        'outcome': list(),
        'prediction': list()
    }
    np.random.seed(42)
    startDate = pd.to_datetime('01-01-2024', dayfirst=True)
    endDate = pd.to_datetime('31-12-2024', dayfirst=True)
    switchDate = pd.to_datetime('01-07-2024', dayfirst=True)
    switchDate2 = pd.to_datetime('01-10-2024', dayfirst=True)
    numdays = (endDate - startDate).days
    switchDays = (switchDate - startDate).days
    switch2Days = (switchDate2 - startDate).days

    # Create some semi-realistic data for n = 50 people
    age = np.random.normal(70, 5, 50)
    systolic_bp = np.random.normal(120, 10, 50)

    for i in range(numdays):
        curday = startDate + dt.timedelta(days=i)
        
        # Generate predictions
        lp = -1.5 + 0.5 * (age-70)/5 + 2 * (systolic_bp - 120)/10
        curpredictions = 1 / (1 + np.exp(-lp))
        if i >= switchDays:
            # Change to incidence rate
            lp = lp + 2.5
        if i >= switch2Days:
            # Change incidence rate again
            lp = lp - 2.5
        
        # Generate outcomes
        curoutcomes = np.random.binomial(1, 1 / (1 + np.exp(-lp)))
            
        # Append to dictionary
        mydict['date'].extend([curday]*50)
        mydict['outcome'].extend(curoutcomes)
        mydict['prediction'].extend(curpredictions)

    df = pd.DataFrame(mydict)

    model = Models.RegularRecalibration()
    model.trigger = Triggers.TimeframeTrigger(model=model, updateTimestep='month', dataStart=df['date'].min(), dataEnd=df['date'].max())
    mytest = PREDICT(data=df, model=model, startDate='min', endDate='max', timestep='week')
    mytest.addLogHook(Accuracy(model))
    mytest.run()

    log = mytest.getLog()
    updatedDates = log['Model Updated']

    # Assuming the model gets checked every week and is updated every month
    trueUpdateDates = {pd.Timestamp('2024-02-05 00:00:00'): True,
        pd.Timestamp('2024-03-04 00:00:00'): True,
        pd.Timestamp('2024-04-01 00:00:00'): True,
        pd.Timestamp('2024-04-29 00:00:00'): True,
        pd.Timestamp('2024-05-27 00:00:00'): True,
        pd.Timestamp('2024-06-24 00:00:00'): True,
        pd.Timestamp('2024-07-22 00:00:00'): True,
        pd.Timestamp('2024-08-19 00:00:00'): True,
        pd.Timestamp('2024-09-16 00:00:00'): True,
        pd.Timestamp('2024-10-14 00:00:00'): True,
        pd.Timestamp('2024-11-11 00:00:00'): True,
        pd.Timestamp('2024-12-09 00:00:00'): True}

    assert updatedDates == trueUpdateDates

    # Check days are 28 days apart
    keys = list(trueUpdateDates.keys())
    differences = [(keys[i] - keys[i - 1]).days for i in range(1, len(keys))]
    consistent = all(x == differences[0] for x in differences)  

    assert consistent
    assert differences[0] == 28 