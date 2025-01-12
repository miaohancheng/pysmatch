import pytest
import numpy as np
import pandas as pd
from pysmatch.utils import (
    drop_static_cols, is_continuous, ks_boot, chi2_distance,
    grouped_permutation_test, std_diff
)

def test_ks_boot():
    tr = np.random.randn(30)
    co = np.random.randn(30) + 1  # 有一定分布差异
    pval = ks_boot(tr, co, nboots=50)  # 小样本就行
    assert 0 <= pval <= 1, "ks_boot pval 应该在 [0, 1] 范围内"

def test_chi2_distance():
    t = np.random.randn(50)
    c = np.random.randn(50) + 0.5
    dist = chi2_distance(t, c, bins=5)
    assert dist >= 0

def test_grouped_permutation_test():
    def my_metric(a, b):
        return np.abs(np.mean(a) - np.mean(b))

    t = np.random.randn(20)
    c = np.random.randn(20)
    p_value, truth = grouped_permutation_test(my_metric, t, c, n_samples=10)
    assert 0 <= p_value <= 1

def test_std_diff():
    a = np.array([1,2,3,4])
    b = np.array([3,4,5,6])
    med_diff, mean_diff = std_diff(a, b)
    # 只要能跑到就行, 具体断言可以更细化
    assert isinstance(med_diff, float)
    assert isinstance(mean_diff, float)