from __future__ import division
import sys
import numpy as np
from scipy import stats


def drop_static_cols(df, yvar, cols=None):
    # Check if the columns list is provided, if not, use all columns in the dataframe
    if cols is None:
        cols = df.columns.tolist()

    # Exclude the target variable column from the list of columns to check for static values
    cols = [col for col in cols if col != yvar]

    # Calculate the number of unique values for each column and identify static columns
    nunique = df[cols].nunique()
    static_cols = nunique[nunique == 1].index.tolist()  # Columns with only one unique value

    # If there are any static columns, drop them from the dataframe
    if static_cols:
        df.drop(columns=static_cols, inplace=True)

        # Print the names of the dropped columns to the standard output with a carriage return to overwrite the last line
        sys.stdout.write('\rStatic columns dropped: {}'.format(', '.join(static_cols)))

    # Return the dataframe with static columns dropped
    return df


def ks_boot(tr, co, nboots=1000):
    # Calculate the length of the treatment group
    nx = len(tr)

    # Combine the treatment and control groups into one array
    w = np.concatenate((tr, co))

    # Calculate the total number of observations in the combined array
    obs = len(w)

    # Set the cutoff point to separate the treatment observations from the control ones in the bootstrap samples
    cutp = nx

    # Compute the Kolmogorov-Smirnov statistic for the original treatment and control samples
    fs_ks, _ = stats.ks_2samp(tr, co)

    # Initialize the counter for tracking the number of bootstrap samples
    # where the computed KS statistic is greater than or equal to the observed KS statistic
    bbcount = 0

    # Perform bootstrap resampling
    for _ in range(nboots):
        # Resample from the combined array with replacement
        sw = np.random.choice(w, obs, replace=True)

        # Split the resample into new treatment and control samples
        x1tmp = sw[:cutp]
        x2tmp = sw[cutp:]

        # Compute the KS statistic for the bootstrap samples
        s_ks, _ = stats.ks_2samp(x1tmp, x2tmp)

        # If the bootstrap KS statistic is greater than or equal to the original KS statistic, increment the count
        if s_ks >= fs_ks:
            bbcount += 1

    # Calculate the bootstrap p-value as the proportion of bootstrap replicates where s_ks >= fs_ks
    ks_boot_pval = bbcount / nboots

    # Return the computed bootstrap p-value
    return ks_boot_pval

def chi2_distance(t, c, bins='auto'):
    # Compute histograms
    t_hist, bin_edges = np.histogram(t, bins=bins)
    c_hist, _ = np.histogram(c, bins=bin_edges)

    # Manually compute the Chi-square distance
    # Avoid division by zero; add a small number (epsilon) to the denominator
    epsilon = 1e-10
    chi2_dist = 0.5 * np.sum(((t_hist - c_hist) ** 2) / (t_hist + c_hist + epsilon))

    return chi2_dist


def grouped_permutation_test(f, t, c, n_samples=1000):
    # Calculate the test statistic for the original groups using the function `f`
    truth = f(t, c)

    # Combine the treatment (t) and control (c) groups into one array for permutation
    comb = np.concatenate((t, c))

    # Store the number of items in the treatment group
    tn = len(t)

    # Initialize the counter for the number of times the permuted statistic is greater than or equal to the original statistic
    times_geq = 0

    # Perform the permutation test for a specified number of samples
    for _ in range(n_samples):
        # Shuffle the combined array to permute the group labels
        np.random.shuffle(comb)

        # Split the shuffled array back into "treatment" and "control" based on original sizes
        tt = comb[:tn]  # Permuted treatment group
        cc = comb[tn:]  # Permuted control group

        # Calculate the test statistic for the permuted groups
        sample_truth = f(tt, cc)

        # Check if the permuted statistic is greater than or equal to the original statistic
        if sample_truth >= truth:
            times_geq += 1

    # Calculate the p-value as the proportion of permutations where the statistic is greater than or equal to the observed
    p_value = times_geq / n_samples

    # Return the p-value and the original test statistic
    return p_value, truth

def std_diff(a, b):
    combined = np.concatenate([a, b])
    sd = np.std(combined, ddof=1)
    if sd == 0:
        return 0, 0
    med_diff = (np.median(a) - np.median(b)) / sd
    mean_diff = (np.mean(a) - np.mean(b)) / sd
    return med_diff, mean_diff


def progress(i, n, prestr=''):
    sys.stdout.write('\r{}: {}/{}'.format(prestr, i, n))
    sys.stdout.flush()


def is_continuous(colname, dmatrix):
    """
    Check if the colname was treated as continuous in the patsy.dmatrix
    Would look like colname[<factor_value>] otherwise
    """
    return (colname in dmatrix.columns) or ("Q('{}')".format(colname) in dmatrix.columns)
