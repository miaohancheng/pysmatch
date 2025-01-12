import pytest
import pandas as pd
import numpy as np

from pysmatch.Matcher import Matcher

@pytest.fixture
def sample_data():
    """
    构造一个简单的测试/对照 DataFrame
    """
    np.random.seed(42)
    size = 50
    test_df = pd.DataFrame({
        'feature1': np.random.randn(size),
        'feature2': np.random.randint(0, 5, size),
        'treatment': 1
    })
    control_df = pd.DataFrame({
        'feature1': np.random.randn(size),
        'feature2': np.random.randint(0, 5, size),
        'treatment': 0
    })
    return test_df, control_df

def test_matcher_init(sample_data):
    """
    测试 Matcher 初始化
    """
    test_df, control_df = sample_data
    matcher = Matcher(test_df, control_df, yvar='treatment')
    assert len(matcher.data) == 100, "合并后的总长度应该是 100"
    assert matcher.yvar == 'treatment'
    assert 'treatment' not in matcher.xvars, "yvar 不应包含在 xvars 中"

def test_matcher_fit_scores(sample_data):
    """
    测试 fit_scores 过程
    """
    test_df, control_df = sample_data
    matcher = Matcher(test_df, control_df, yvar='treatment')
    matcher.fit_scores(balance=True, model_type='linear', n_jobs=1)
    assert len(matcher.models) > 0, "至少要有一个模型被训练"
    assert len(matcher.model_accuracy) == len(matcher.models), "模型准确率列表长度要与模型列表相同"

def test_matcher_predict_scores(sample_data):
    """
    测试 predict_scores
    """
    test_df, control_df = sample_data
    matcher = Matcher(test_df, control_df, yvar='treatment')
    matcher.fit_scores(balance=False, model_type='linear')
    matcher.predict_scores()
    assert 'scores' in matcher.data.columns, "预测完成后应在 data 中生成 'scores' 列"
    assert not matcher.data['scores'].isnull().any(), "scores 列中不应该存在空值"

def test_matcher_match(sample_data):
    """
    测试 match 方法
    """
    test_df, control_df = sample_data
    matcher = Matcher(test_df, control_df, yvar='treatment')
    matcher.fit_scores()
    matcher.predict_scores()
    matcher.match(threshold=0.01)
    # 看看 matched_data 是否非空
    assert len(matcher.matched_data) > 0, "匹配结果应当有数据"