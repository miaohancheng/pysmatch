import pytest
import pandas as pd
import numpy as np
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

def test_fit_model(sample_xy):
    X, y = sample_xy
    result = fit_model(index=0, X=X, y=y, model_type='linear', balance=False)
    model = result['model']
    accuracy = result['accuracy']
    assert model is not None, "应该返回训练后的模型"
    assert 0 <= accuracy <= 1, "accuracy 应该在 [0,1] 之间"

def test_optuna_tuner(sample_xy):
    X, y = sample_xy
    result = optuna_tuner(X, y, model_type='linear', n_trials=2, balance=False)
    model = result['model']
    best_score = result['accuracy']
    assert model is not None, "Optuna 调参后也应返回最终的模型"
    assert 0 <= best_score <= 1, "best_score 应该在 [0,1] 之间"