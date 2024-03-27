import pytest
import pandas as pd

from par_coordinates import get_results

# these tests are all quite trivial and based around creating a mocked tuner for the hyperparameter search
# for completeness, an example is added for get_results.sklearn
@pytest.fixture
def mocked_tuner_sklearn(mocker):
    # create a mocked tuner object
    tuner = mocker.MagicMock()
    tuner.cv_results_ = {
        "mean_test_score": [0.9, 0.8, 0.7],
        "params": [{"A": 1, "B": "1"},
                   {"A": 2, "B": "2"},
                   {"A": 3, "B": "3"}]
    }
    return tuner

def test_keras_tuner():
    pass

def test_sklearn(mocked_tuner_sklearn):
    results = get_results.sklearn(mocked_tuner_sklearn, "mse")
    assert results.equals(pd.DataFrame([{"A": 1, "B": "1", "mse": 0.9},
                                        {"A": 2, "B": "2", "mse": 0.8},
                                        {"A": 3, "B": "3", "mse": 0.7}]))

def test_optuna():
    pass

