import pytest
import pandas as pd
import numpy as np
from pysmatch.visualization import plot_scores

def test_plot_scores():
    data = pd.DataFrame({
        'scores': np.random.rand(10),
        'yvar': [0,1,0,1,0,1,0,1,0,1]
    })
    # 这里可以用 pytest-mpl 或者一些其他替代方案做可视化测试
    # 简单测试只检查是否会抛异常
    try:
        plot_scores(data, yvar='yvar')
    except Exception as e:
        pytest.fail(f"plot_scores 执行报错: {e}")