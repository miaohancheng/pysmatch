import pytest
import pandas as pd
import numpy as np
from unittest import mock
from pysmatch.visualization import plot_scores

@mock.patch("matplotlib.pyplot.show")
def test_plot_scores_no_error(_mock_show):
    data = pd.DataFrame({
        'scores': np.random.rand(10),
        'yvar': [0,1,0,1,0,1,0,1,0,1]
    })
    try:
        plot_scores(data, yvar='yvar')
    except Exception as e:
        pytest.fail(f"plot_scores 执行报错: {e}")
