# utils.py
# -*- coding: utf-8 -*-
import sys
import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, List, Union

def drop_static_cols(df: pd.DataFrame, yvar: str, cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Drops static columns (columns with only one unique value) from a DataFrame.
    中文注释: 删除只有单一取值的列
    """
    if cols is None:
        cols = df.columns.tolist()
    cols = [col for col in cols if col != yvar]
    nunique = df[cols].nunique()
    static_cols = nunique[nunique == 1].index.tolist()
    if static_cols:
        df.drop(columns=static_cols, inplace=True)
        sys.stdout.write(f'\rStatic columns dropped: {", ".join(static_cols)}')
        sys.stdout.flush()
    return df

def ks_boot(tr: np.ndarray, co: np.ndarray, nboots: int = 1000) -> float:
    """
    Performs a bootstrap Kolmogorov-Smirnov test to calculate the p-value.
    中文注释: 通过自举法来计算 KS 检验的 p-value
    """
    nx = len(tr)
    combined = np.concatenate((tr, co))
    obs = len(combined)
    fs_ks, _ = stats.ks_2samp(tr, co)

    bbcount = 0
    for _ in range(nboots):
        sample = np.random.choice(combined, obs, replace=True)
        x1 = sample[:nx]
        x2 = sample[nx:]
        s_ks, _ = stats.ks_2samp(x1, x2)
        if s_ks >= fs_ks:
            bbcount += 1
    return bbcount / nboots

def chi2_distance(t: np.ndarray, c: np.ndarray, bins: Union[int, str] = 'auto') -> float:
    """
    Computes the Chi-square distance between two distributions.
    中文注释: 计算卡方距离
    """
    t_hist, bin_edges = np.histogram(t, bins=bins)
    c_hist, _ = np.histogram(c, bins=bin_edges)
    epsilon = 1e-10
    return 0.5 * np.sum(((t_hist - c_hist) ** 2) / (t_hist + c_hist + epsilon))

def grouped_permutation_test(f, t: np.ndarray, c: np.ndarray, n_samples: int = 1000) -> tuple:
    """
    Performs a grouped permutation test to evaluate the significance of a test statistic.
    中文注释: 分组置换检验
    """
    truth = f(t, c)
    combined = np.concatenate((t, c))
    tn = len(t)
    count = 0
    for _ in range(n_samples):
        np.random.shuffle(combined)
        tt = combined[:tn]
        cc = combined[tn:]
        sample_truth = f(tt, cc)
        if sample_truth >= truth:
            count += 1
    p_value = count / n_samples
    return p_value, truth

def std_diff(a: np.ndarray, b: np.ndarray) -> tuple:
    """
    Calculates the standardized median and mean differences between two groups.
    中文注释: 计算两个组间的标准化中位数和均值差异
    """
    combined = np.concatenate([a, b])
    sd = np.std(combined, ddof=1)
    if sd == 0:
        return 0, 0
    med_diff = (np.median(a) - np.median(b)) / sd
    mean_diff = (np.mean(a) - np.mean(b)) / sd
    return med_diff, mean_diff

def progress(i: int, n: int, prestr: str = '') -> None:
    """
    Displays the current progress of a process in the console.
    中文注释: 打印进度条
    """
    sys.stdout.write(f'\r{prestr}: {i}/{n}')
    sys.stdout.flush()

def is_continuous(colname: str, df: pd.DataFrame) -> bool:
    """
    Checks if 'colname' is numeric in 'df'.
    中文注释: 判断是否连续性变量
    """
    if colname not in df.columns:
        return False
    return pd.api.types.is_numeric_dtype(df[colname])