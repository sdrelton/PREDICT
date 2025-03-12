import pandas as pd
import pytest
from datetime import datetime, timedelta
import PREDICT.PREDICT as PREDICT


class MockModel:
    """Mock model for testing the PREDICT class.
    """
    def predict(self, data):
        pass
    def trigger(self, data):
        return True
    def update(self, data):
        pass

@pytest.fixture
def sample_data():
    """Creates a sample dataset for testing the PREDICT class.

    Returns:
        pd.DataFrame: Fake dataset with date and value columns.
    """
    data = {
        'date': pd.date_range(start='1/1/2020', periods=10, freq='D'),
        'value': range(10)
    }
    return pd.DataFrame(data)

@pytest.fixture
def predict_instance(sample_data):
    """Creates an instance of the PREDICT class using the provided sample data.

    Args:
        sample_data (pd.DataFrame): Sample data to be used for creating the PREDICT instance.

    Returns:
        An instance of the PREDICT class.
    """
    model = MockModel()
    return PREDICT(sample_data, model)

def test_initialisation(predict_instance):
    """Test that the PREDICT class is initialised correctly.

    Args:
        predict_instance (class): Initialises with a dataset and a prediction model. 
        Manages prediction windows and logs, allows for the addition of hooks to execute functions 
        during the logging process.
    """
    assert predict_instance.startDate == predict_instance.data['date'].min()
    assert predict_instance.endDate == predict_instance.data['date'].max()
    assert predict_instance.timestep == pd.Timedelta(weeks=1)
    assert predict_instance.currentWindowStart == predict_instance.startDate
    assert predict_instance.currentWindowEnd == predict_instance.startDate + predict_instance.timestep
    assert predict_instance.log == {}
    assert predict_instance.logHooks == []

def test_add_log_hook(predict_instance):
    """Test that hooks can be added to the log.

    Args:
        predict_instance (class): Initialises with a dataset and a prediction model. 
        Manages prediction windows and logs, allows for the addition of hooks to execute functions 
        during the logging process.
    """
    hook = lambda data: ('hook_name', 'result')
    predict_instance.addLogHook(hook)
    assert hook in predict_instance.logHooks

def test_add_log(predict_instance):
    """Test that logs can be added to the log dictionary.

    Args:
        predict_instance (class): Initialises with a dataset and a prediction model. 
        Manages prediction windows and logs, allows for the addition of hooks to execute functions 
        during the logging process.
    """
    predict_instance.addLog('test', '2020-01-01', 'value')
    assert 'test' in predict_instance.log
    assert '2020-01-01' in predict_instance.log['test']
    assert predict_instance.log['test']['2020-01-01'] == 'value'

def test_get_log(predict_instance):
    """Test that the getLog method returns the log dictionary.

    Args:
        predict_instance (class): Initialises with a dataset and a prediction model. 
        Manages prediction windows and logs, allows for the addition of hooks to execute functions 
        during the logging process.
    """
    predict_instance.addLog('test', '2020-01-01', 'value')
    log = predict_instance.getLog()
    assert log == {'test': {'2020-01-01': 'value'}}

def test_run(predict_instance):
    """Test that the run method executes the prediction model.

    Args:
        predict_instance (class): Initialises with a dataset and a prediction model. 
        Manages prediction windows and logs, allows for the addition of hooks to execute functions 
        during the logging process.
    """
    
    def log_hook(data):
        return 'log_hook', len(data)

    predict_instance.addLogHook(log_hook)
    predict_instance.run()

    # Check if logs were added
    assert 'log_hook' in predict_instance.log
    assert len(predict_instance.log['log_hook']) > 0
    assert 'Model Updated' in predict_instance.log
    assert len(predict_instance.log['Model Updated']) > 0
