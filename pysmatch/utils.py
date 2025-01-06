from __future__ import division
import sys
import numpy as np
from scipy import stats
import pandas as pd


def drop_static_cols(df: pd.DataFrame, yvar: str, cols: list = None) -> pd.DataFrame:
    """
    Drops static columns (columns with only one unique value) from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame from which to drop static columns.
    yvar : str
        The name of the target variable column to exclude from dropping.
    cols : list of str, optional
        A list of columns to check for static values. If not provided, all columns in the DataFrame are checked.

    Returns
    -------
    pd.DataFrame
        A DataFrame with static columns removed.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'A': [1, 1, 1],
    ...     'B': [2, 3, 4],
    ...     'Y': [0, 1, 0]
    ... })
    >>> drop_static_cols(df, yvar='Y')
       B  Y
    0  2  0
    1  3  1
    2  4  0
    """
    # If no specific columns are provided, use all columns in the DataFrame
    if cols is None:
        cols = df.columns.tolist()

    # Exclude the target variable column from the list of columns to check for static values
    cols = [col for col in cols if col != yvar]

    # Calculate the number of unique values for each column and identify static columns
    nunique = df[cols].nunique()
    static_cols = nunique[nunique == 1].index.tolist()  # Columns with only one unique value

    # If there are any static columns, drop them from the DataFrame
    if static_cols:
        df.drop(columns=static_cols, inplace=True)

        # Print the names of the dropped columns to the standard output with a carriage return to overwrite the last line
        sys.stdout.write('\rStatic columns dropped: {}'.format(', '.join(static_cols)))
        sys.stdout.flush()

    # Return the DataFrame with static columns dropped
    return df


def ks_boot(tr: np.ndarray, co: np.ndarray, nboots: int = 1000) -> float:
    """
    Performs a bootstrap Kolmogorov-Smirnov test to calculate the p-value.

    Parameters
    ----------
    tr : np.ndarray
        The treatment group data.
    co : np.ndarray
        The control group data.
    nboots : int, optional
        The number of bootstrap samples to generate, by default 1000.

    Returns
    -------
    float
        The bootstrap p-value indicating the significance of the KS statistic.

    Examples
    --------
    >>> tr = np.array([1, 2, 3, 4, 5])
    >>> co = np.array([2, 3, 4, 5, 6])
    >>> ks_boot(tr, co, nboots=1000)
    0.04
    """
    # Number of samples in the treatment group
    nx = len(tr)

    # Combine treatment and control groups into one array
    w = np.concatenate((tr, co))

    # Total number of observations
    obs = len(w)

    # Set the cutoff point to separate treatment and control samples in bootstrap
    cutp = nx

    # Compute the Kolmogorov-Smirnov statistic for the original samples
    fs_ks, _ = stats.ks_2samp(tr, co)

    # Initialize counter for bootstrap samples where KS statistic >= original KS statistic
    bbcount = 0

    # Perform bootstrap resampling
    for _ in range(nboots):
        # Resample with replacement from the combined array
        sw = np.random.choice(w, obs, replace=True)

        # Split the resampled array back into treatment and control samples
        x1tmp = sw[:cutp]
        x2tmp = sw[cutp:]

        # Compute the KS statistic for the bootstrap samples
        s_ks, _ = stats.ks_2samp(x1tmp, x2tmp)

        # Increment the counter if the bootstrap KS statistic is >= original KS statistic
        if s_ks >= fs_ks:
            bbcount += 1

    # Calculate the bootstrap p-value
    ks_boot_pval = bbcount / nboots

    return ks_boot_pval


def chi2_distance(t: np.ndarray, c: np.ndarray, bins: int = 'auto') -> float:
    """
    Computes the Chi-square distance between two distributions.

    Parameters
    ----------
    t : np.ndarray
        The treatment group data.
    c : np.ndarray
        The control group data.
    bins : int, sequence of scalars, or str, optional
        The number of bins or bin edges for the histogram, by default 'auto'.

    Returns
    -------
    float
        The calculated Chi-square distance between the two distributions.

    Examples
    --------
    >>> t = np.array([1, 2, 3, 4, 5])
    >>> c = np.array([2, 3, 4, 5, 6])
    >>> chi2_distance(t, c, bins=5)
    0.1
    """
    # Compute histograms for treatment and control groups
    t_hist, bin_edges = np.histogram(t, bins=bins)
    c_hist, _ = np.histogram(c, bins=bin_edges)

    # Manually compute the Chi-square distance
    # Avoid division by zero by adding a small epsilon to the denominator
    epsilon = 1e-10
    chi2_dist = 0.5 * np.sum(((t_hist - c_hist) ** 2) / (t_hist + c_hist + epsilon))

    return chi2_dist


def grouped_permutation_test(f, t: np.ndarray, c: np.ndarray, n_samples: int = 1000) -> tuple:
    """
    Performs a grouped permutation test to evaluate the significance of a test statistic.

    Parameters
    ----------
    f : callable
        A function that computes the test statistic. It should accept two parameters (treatment and control groups).
    t : np.ndarray
        The treatment group data.
    c : np.ndarray
        The control group data.
    n_samples : int, optional
        The number of permutation samples to generate, by default 1000.

    Returns
    -------
    tuple
        A tuple containing the p-value and the original test statistic (p_value, truth).

    Examples
    --------
    >>> def test_stat(a, b):
    ...     return np.mean(a) - np.mean(b)
    >>> t = np.array([1, 2, 3, 4, 5])
    >>> c = np.array([2, 3, 4, 5, 6])
    >>> grouped_permutation_test(test_stat, t, c, n_samples=1000)
    (0.04, 0.0)
    """
    # Calculate the test statistic for the original groups using function `f`
    truth = f(t, c)

    # Combine the treatment and control groups into one array for permutation
    comb = np.concatenate((t, c))
    tn = len(t)  # Number of samples in the treatment group

    # Initialize counter for permutation samples with statistic >= original statistic
    times_geq = 0

    # Perform the permutation test
    for _ in range(n_samples):
        # Shuffle the combined array to permute group labels
        np.random.shuffle(comb)

        # Split the shuffled array back into treatment and control groups
        tt = comb[:tn]
        cc = comb[tn:]

        # Calculate the test statistic for the permuted groups
        sample_truth = f(tt, cc)

        # Increment the counter if permuted statistic >= original statistic
        if sample_truth >= truth:
            times_geq += 1

    # Calculate the p-value as the proportion of permutations where statistic >= original
    p_value = times_geq / n_samples

    return p_value, truth


def std_diff(a: np.ndarray, b: np.ndarray) -> tuple:
    """
    Calculates the standardized median and mean differences between two groups.

    Parameters
    ----------
    a : np.ndarray
        The first group data.
    b : np.ndarray
        The second group data.

    Returns
    -------
    tuple
        A tuple containing the standardized median difference and standardized mean difference (med_diff, mean_diff).

    Examples
    --------
    >>> a = np.array([1, 2, 3, 4, 5])
    >>> b = np.array([2, 3, 4, 5, 6])
    >>> std_diff(a, b)
    (0.0, 0.0)
    """
    # Combine both groups to calculate the overall standard deviation
    combined = np.concatenate([a, b])
    sd = np.std(combined, ddof=1)

    # If the standard deviation is zero, return zero differences
    if sd == 0:
        return 0, 0

    # Calculate the standardized median and mean differences
    med_diff = (np.median(a) - np.median(b)) / sd
    mean_diff = (np.mean(a) - np.mean(b)) / sd

    return med_diff, mean_diff


def progress(i: int, n: int, prestr: str = '') -> None:
    """
    Displays the current progress of a process in the console.

    Parameters
    ----------
    i : int
        The current step number.
    n : int
        The total number of steps.
    prestr : str, optional
        A prefix string to display before the progress (e.g., "Processing"), by default ''.

    Returns
    -------
    None

    Examples
    --------
    for i in range(100):
         progress(i, 100, prestr='Processing')
    """
    # Write the progress to the standard output, overwriting the previous line
    sys.stdout.write('\r{}: {}/{}'.format(prestr, i, n))
    sys.stdout.flush()


def is_continuous(colname: str, dmatrix: pd.DataFrame) -> bool:
    """
    Checks if a column is treated as continuous in the design matrix.

    Parameters
    ----------
    colname : str
        The name of the column to check.
    dmatrix : pd.DataFrame
        The design matrix generated by patsy.dmatrices.

    Returns
    -------
    bool
        True if the column is treated as continuous, False otherwise.
    """
    # Check if the column name exists directly or is wrapped with Q() in the design matrix
    return (colname in dmatrix.columns) or (f"Q('{colname}')" in dmatrix.columns)