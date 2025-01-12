import pytest
import numpy as np
import pandas as pd
from pysmatch.modeling import fit_model, optuna_tuner

@pytest.fixture
def sample_xy():
    np.random.seed(42)
    size = 100
    X = pd.DataFrame({
        'num_feature': np.random.randn(size),
        'cat_feature': np.random.choice(['A','B','C'], size=size)
    })
    # 二分类标签
    y = (np.random.rand(size) > 0.5).astype(int)
    return X, y

@pytest.mark.parametrize("model_type", ["linear", "tree", "knn"])
def test_fit_model_multi_type(sample_xy, model_type):
    """
    同时测试三种模型: logistic, catboost tree, knn
    """
    X, y = sample_xy
    result = fit_model(index=0, X=X, y=y, model_type=model_type, balance=True, max_iter=50)
    model = result['model']
    accuracy = result['accuracy']
    assert model is not None, f"{model_type} 应该返回训练后的模型"
    assert 0 <= accuracy <= 1

@pytest.mark.parametrize("model_type", ["linear", "tree", "knn"])
def test_optuna_tuner_multi_type(sample_xy, model_type):
    """
    使用 optuna_tuner 测试三种模型类型, n_trials=2 以节约时间
    """
    X, y = sample_xy
    result = optuna_tuner(X, y, model_type=model_type, n_trials=2, balance=True)
    model = result['model']
    best_score = result['accuracy']
    assert model is not None, "Optuna 调参后应返回最终的模型"
    assert 0 <= best_score <= 1