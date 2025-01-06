# -*- coding: utf-8 -*-
from __future__ import division
import sys
import numpy as np
from scipy import stats
import pandas as pd


def drop_static_cols(df: pd.DataFrame, yvar: str, cols: list = None) -> pd.DataFrame:
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
        sys.stdout.write('\rStatic columns dropped: {}'.format(', '.join(static_cols)))
        sys.stdout.flush()
    return df


def ks_boot(tr: np.ndarray, co: np.ndarray, nboots: int = 1000) -> float:
    """
    Performs a bootstrap Kolmogorov-Smirnov test to calculate the p-value.
    中文注释: 通过自举法来计算 KS 检验的 p-value
    """
    nx = len(tr)
    w = np.concatenate((tr, co))
    obs = len(w)
    cutp = nx
    fs_ks, _ = stats.ks_2samp(tr, co)

    bbcount = 0
    for _ in range(nboots):
        sw = np.random.choice(w, obs, replace=True)
        x1tmp = sw[:cutp]
        x2tmp = sw[cutp:]
        s_ks, _ = stats.ks_2samp(x1tmp, x2tmp)
        if s_ks >= fs_ks:
            bbcount += 1
    ks_boot_pval = bbcount / nboots
    return ks_boot_pval


def chi2_distance(t: np.ndarray, c: np.ndarray, bins: int = 'auto') -> float:
    """
    Computes the Chi-square distance between two distributions.
    中文注释: 计算卡方距离
    """
    t_hist, bin_edges = np.histogram(t, bins=bins)
    c_hist, _ = np.histogram(c, bins=bin_edges)
    epsilon = 1e-10
    chi2_dist = 0.5 * np.sum(((t_hist - c_hist) ** 2) / (t_hist + c_hist + epsilon))
    return chi2_dist


def grouped_permutation_test(f, t: np.ndarray, c: np.ndarray, n_samples: int = 1000) -> tuple:
    """
    Performs a grouped permutation test to evaluate the significance of a test statistic.
    中文注释: 分组置换检验
    """
    truth = f(t, c)
    comb = np.concatenate((t, c))
    tn = len(t)

    times_geq = 0
    for _ in range(n_samples):
        np.random.shuffle(comb)
        tt = comb[:tn]
        cc = comb[tn:]
        sample_truth = f(tt, cc)
        if sample_truth >= truth:
            times_geq += 1

    p_value = times_geq / n_samples
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
    sys.stdout.write('\r{}: {}/{}'.format(prestr, i, n))
    sys.stdout.flush()


def is_continuous(colname: str, dmatrix: pd.DataFrame) -> bool:
    """
    Checks if a column is treated as continuous in the design matrix.
    中文注释: 判断某列在设计矩阵中是否被视为连续变量
    """
    return (colname in dmatrix.columns) or (f"Q('{colname}')" in dmatrix.columns)