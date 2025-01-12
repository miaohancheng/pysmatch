import pytest
import numpy as np
import pandas as pd

from pysmatch.utils import drop_static_cols, is_continuous

def test_drop_static_cols():
    df = pd.DataFrame({
        'col1': [1,1,1],  # static
        'col2': [1,2,3],  # not static
        'target': [0,1,0]
    })
    new_df = drop_static_cols(df, yvar='target')
    assert 'col1' not in new_df.columns, "static 列应当被删除"
    assert 'col2' in new_df.columns

def test_is_continuous():
    df = pd.DataFrame({
        "Q('feature1')": [1.0, 2.2, 3.5],
        "Q('feature2')": [2, 4, 6]
    })
    # 即使列名加了 Q('') 这种，函数里也会判断是不是出现在设计矩阵中
    assert is_continuous("feature1", df) is True
    assert is_continuous("feature2", df) is True
    assert is_continuous("feature3", df) is False