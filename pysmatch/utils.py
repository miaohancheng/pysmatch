# utils.py
# -*- coding: utf-8 -*-
import sys
import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, List, Union

def drop_static_cols(df: pd.DataFrame, yvar: str, cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Drops columns from a DataFrame that contain only a single unique value (static columns).

    It identifies columns with only one unique value among the specified `cols` (excluding `yvar`)
    and removes them in place from the DataFrame. Prints the names of dropped columns.

    Args:
        df (pd.DataFrame): The input DataFrame to modify.
        yvar (str): The name of the target variable column, which should not be dropped even if static.
        cols (Optional[List[str]], optional): A list of column names to check for static values.
                                              If None, checks all columns in the DataFrame
                                              (excluding `yvar`). Defaults to None.

    Returns:
        pd.DataFrame: The input DataFrame `df` modified in place (static columns removed).
                      It returns the same DataFrame object that was passed in.
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
    Performs a bootstrap Kolmogorov-Smirnov (KS) test to estimate the p-value.

    This function estimates the p-value for the two-sample KS test by comparing the
    observed KS statistic between the two input samples (`tr` and `co`) against a
    distribution of KS statistics obtained from bootstrap samples drawn under the null
    hypothesis (that both samples come from the same distribution).

    Args:
        tr (np.ndarray): Array containing data for the first sample (e.g., treatment group).
        co (np.ndarray): Array containing data for the second sample (e.g., control group).
        nboots (int, optional): The number of bootstrap iterations to perform.
                                Defaults to 1000.

    Returns:
        float: The estimated p-value, calculated as the proportion of bootstrap KS statistics
               that are greater than or equal to the observed KS statistic.
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
    Computes the Chi-square distance between the distributions of two samples.

    This function calculates a measure of distance between the histograms of two
    samples (`t` and `c`). It first creates histograms with common bins, then
    computes the Chi-square statistic based on the frequencies in each bin.
    A small epsilon is added to the denominator to avoid division by zero.

    Args:
        t (np.ndarray): Array containing data for the first sample.
        c (np.ndarray): Array containing data for the second sample.
        bins (Union[int, str], optional): The number of bins or the binning strategy
                                          to use for `np.histogram`. Defaults to 'auto'.

    Returns:
        float: The calculated Chi-square distance. Returns 0.0 if inputs are empty
               or identical after binning.
    """
    t_hist, bin_edges = np.histogram(t, bins=bins)
    c_hist, _ = np.histogram(c, bins=bin_edges)
    epsilon = 1e-10
    return 0.5 * np.sum(((t_hist - c_hist) ** 2) / (t_hist + c_hist + epsilon))

def grouped_permutation_test(f, t: np.ndarray, c: np.ndarray, n_samples: int = 1000) -> tuple:
    """
    Performs a permutation test for a given statistic function `f`.

    Evaluates the significance of an observed test statistic calculated by function `f`
    applied to samples `t` and `c`. It does this by:
    1. Calculating the observed statistic `truth = f(t, c)`.
    2. Repeatedly (`n_samples` times):
        a. Combining `t` and `c`.
        b. Shuffling the combined data.
        c. Splitting the shuffled data back into two samples of the original sizes.
        d. Calculating the statistic `f` on the permuted samples.
        e. Counting how often the permuted statistic is greater than or equal to `truth`.
    3. Calculating the p-value as the proportion of permuted statistics that met the condition in 2e.

    Args:
        f (Callable[[np.ndarray, np.ndarray], float]): A function that takes two numpy arrays
                                                        (representing two groups) and returns a
                                                        single float value (the test statistic).
        t (np.ndarray): Array containing data for the first group.
        c (np.ndarray): Array containing data for the second group.
        n_samples (int, optional): The number of permutation iterations to perform.
                                   Defaults to 1000.

    Returns:
        tuple: A tuple containing:
            - p_value (float): The estimated p-value from the permutation test.
            - truth (float): The observed test statistic calculated on the original samples.
                             Returns (1.0, np.nan) or similar if inputs are invalid.
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
    Calculates standardized differences (median and mean) between two arrays.

    Computes the difference in medians and means between arrays `a` and `b`,
    standardized by the pooled standard deviation of the combined data.

    Args:
        a (np.ndarray): Array containing data for the first group.
        b (np.ndarray): Array containing data for the second group.

    Returns:
        Tuple[float, float]: A tuple containing:
            - med_diff (float): The standardized median difference `(median(a) - median(b)) / std(combined)`.
            - mean_diff (float): The standardized mean difference `(mean(a) - mean(b)) / std(combined)`.
                                Returns (0.0, 0.0) if the combined standard deviation is zero or
                                if inputs are empty.
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
    Displays a simple progress indicator in the console.

    Prints a progress string like "[prefix]: i/n" to standard output, overwriting
    the previous line using the carriage return character `\r`.

    Args:
        i (int): The current step or item number (should be 1-based or adjusted).
        n (int): The total number of steps or items.
        prestr (str, optional): A prefix string to display before the progress count
                                (e.g., "Processing file"). Defaults to ''.
    """
    sys.stdout.write(f'\r{prestr}: {i}/{n}')
    sys.stdout.flush()

def is_continuous(colname: str, df: pd.DataFrame) -> bool:
    """
    Checks if a specified column in a DataFrame has a numeric data type.

    Uses `pandas.api.types.is_numeric_dtype` to determine if the column
    contains numerical data (integers or floats).

    Args:
        colname (str): The name of the column to check.
        df (pd.DataFrame): The DataFrame containing the column.

    Returns:
        bool: True if the column exists in the DataFrame and its dtype is numeric,
              False otherwise (column doesn't exist or dtype is non-numeric).
    """
    if colname not in df.columns:
        return False
    return pd.api.types.is_numeric_dtype(df[colname])