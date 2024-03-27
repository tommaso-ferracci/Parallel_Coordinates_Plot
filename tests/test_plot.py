import pytest
import numpy as np
import pandas as pd

from par_coordinates import plot

@pytest.fixture()
def some_data():
    # example dataframe to be used for testing
    data = {"A": [1, 2, 3, 4, 5],
            "B": ["1", "2", "3", "4", "5"],
            "C": [1., 2., 3., 4., 5.]}
    data = pd.DataFrame(data)
    return data

def test_set_ytype(some_data):
    ytype = plot.set_ytype(some_data)
    assert ytype == ["linear", "factor", "linear"]

def test_set_ylabels(some_data):
    ytype = ["linear", "factor", "linear"]
    ylabels = plot.set_ylabels(some_data, ytype)
    assert ylabels == [[], ["1", "2", "3", "4", "5"], []]

def test_replace_str_values(some_data):
    ytype = ["linear", "factor", "linear"]
    ylabels = [[], ["1", "2", "3", "4", "5"], []]
    array = plot.replace_str_values(some_data, ytype, ylabels)
    assert (array == np.array([[1, 2, 3, 4, 5],
                               [0, 1, 2, 3, 4],
                               [1, 2, 3, 4, 5]])).all()
    
def test_set_ylim(some_data):
    ytype = ["linear", "factor", "linear"]
    ylabels = [[], ["1", "2", "3", "4", "5"], []]
    array = plot.replace_str_values(some_data, ytype, ylabels)
    ylim = plot.set_ylim(array)
    assert ylim == [[1, 5], [0, 4], [1., 5.]]

def test_get_performance(some_data):
    ytype = ["linear", "factor", "linear"]
    ylabels = [[], ["1", "2", "3", "4", "5"], []]
    array = plot.replace_str_values(some_data, ytype, ylabels)
    ylim = plot.set_ylim(array)
    performance = plot.get_performance(array, ylim)
    assert (performance == np.array([0, 0.25, 0.5, 0.75, 1])).all()

def test_rescale_data(some_data):
    ytype = ["linear", "factor", "linear"]
    ylabels = [[], ["1", "2", "3", "4", "5"], []]
    array = plot.replace_str_values(some_data, ytype, ylabels)
    ylim = plot.set_ylim(array)
    scaled_array = plot.rescale_data(array, ytype, ylim)
    assert (scaled_array == np.array([[1, 2, 3, 4, 5],
                                      [1, 2, 3, 4, 5],
                                      [1, 2, 3, 4, 5]])).all()
    
