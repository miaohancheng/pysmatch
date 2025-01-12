import pytest
import numpy as np
import pandas as pd
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

def test_fit_scores_optuna(sample_data):
    """
    测试 fit_scores(use_optuna=True) 分支, 覆盖 Optuna Tuner
    """
    test_df, control_df = sample_data
    matcher = Matcher(test_df, control_df, yvar='treatment')
    matcher.fit_scores(use_optuna=True, model_type='linear', n_trials=1)  # 只跑1个trial省时间
    assert len(matcher.models) == 1
    assert len(matcher.model_accuracy) == 1
    assert matcher.model_type == 'linear'

def test_fit_scores_multi_model(sample_data):
    """
    测试 fit_scores 传入 nmodels>1 的分支, 走并行逻辑
    """
    test_df, control_df = sample_data
    matcher = Matcher(test_df, control_df, yvar='treatment')
    # 对小数据集可能要减少 n_jobs 或 nmodels
    matcher.fit_scores(balance=True, model_type='linear', nmodels=2, n_jobs=1)
    assert len(matcher.models) == 2
    assert len(matcher.model_accuracy) == 2

def test_match_replacement(sample_data):
    """
    测试 match 时使用 replacement=True
    """
    test_df, control_df = sample_data
    matcher = Matcher(test_df, control_df, yvar='treatment')
    matcher.fit_scores()
    matcher.predict_scores()
    matcher.match(threshold=0.01, replacement=True)
    assert len(matcher.matched_data) > 0, "匹配结果不应为空"
    # 验证 replacement 一般会产生更多匹配记录

def test_tune_threshold(sample_data):
    """
    测试 tune_threshold 方法
    """
    test_df, control_df = sample_data
    matcher = Matcher(test_df, control_df, yvar='treatment')
    matcher.fit_scores()
    matcher.predict_scores()
    # 这里 range 可稍微扩大些，当然要考虑运行时间
    rng = np.arange(0, 0.005, 0.001)
    matcher.tune_threshold(method='min', nmatches=1, rng=rng)
    # 单纯保证不报错即可；实际可细化断言

def test_record_frequency_and_weights(sample_data):
    """
    测试 record_frequency 和 assign_weight_vector
    """
    test_df, control_df = sample_data
    matcher = Matcher(test_df, control_df, yvar='treatment')
    matcher.fit_scores()
    matcher.predict_scores()
    matcher.match(threshold=0.01)
    freq_df = matcher.record_frequency()
    assert isinstance(freq_df, pd.DataFrame)
    matcher.assign_weight_vector()
    # assign_weight_vector 后 matched_data 应多出 weight 列
    assert 'weight' in matcher.matched_data.columns

def test_prop_test(sample_data):
    """
    测试 prop_test 对某个离散变量做卡方检验
    """
    test_df, control_df = sample_data
    # 人工加入个离散变量
    test_df['cat_var'] = np.random.choice(['A', 'B'], size=len(test_df))
    control_df['cat_var'] = np.random.choice(['A', 'B'], size=len(control_df))
    matcher = Matcher(test_df, control_df, yvar='treatment')
    matcher.fit_scores()
    matcher.predict_scores()
    matcher.match(threshold=0.01)
    res = matcher.prop_test('cat_var')
    # 结果应该包含 before/after p值
    assert res is not None
    assert 'before' in res and 'after' in res

def test_compare_continuous(sample_data):
    """
    测试 compare_continuous 分支
    """
    test_df, control_df = sample_data
    matcher = Matcher(test_df, control_df, yvar='treatment')
    matcher.fit_scores()
    matcher.predict_scores()
    matcher.match(threshold=0.01)
    # 注意: matched_data 不为空时，才能compare
    # 如果想避免弹出画图窗口, 可以 set plot_result=False
    df_result = matcher.compare_continuous(return_table=True, plot_result=False)
    # 检查结果是否为 DataFrame
    assert df_result is not None

def test_compare_categorical(sample_data):
    """
    测试 compare_categorical 分支
    """
    test_df, control_df = sample_data
    # 手动加个分类变量
    test_df['cat_var'] = np.random.choice(['X', 'Y'], size=len(test_df))
    control_df['cat_var'] = np.random.choice(['X', 'Y'], size=len(control_df))
    matcher = Matcher(test_df, control_df, yvar='treatment')
    matcher.fit_scores()
    matcher.predict_scores()
    matcher.match(threshold=0.01)
    df_cat_res = matcher.compare_categorical(return_table=True, plot_result=False)
    assert df_cat_res is not None
    # 结果里应包含 'var', 'before', 'after'
    if not df_cat_res.empty:
        assert all(col in df_cat_res.columns for col in ['var','before','after'])