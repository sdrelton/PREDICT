import numpy as np
from PREDICT import Models

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