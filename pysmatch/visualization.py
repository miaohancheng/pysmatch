# visualization.py
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from typing import Optional
from pysmatch import utils as uf



def plot_matched_scores(data: pd.DataFrame, yvar: str, control_color: str = "#1F77B4", test_color: str = "#FF7F0E") -> None:
    """
    Plots the distribution of propensity scores after matching.

    Generates Kernel Density Estimate (KDE) plots to visualize the overlap
    of propensity scores between the test and control groups present in the
    *matched* dataset.

    Args:
        data (pd.DataFrame): The matched DataFrame, must contain the `yvar`
                             column and a 'scores' column.
        yvar (str): The name of the binary column indicating group membership (0 or 1).
        control_color (str, optional): Hex color code for the control group plot.
                                       Defaults to "#1F77B4".
        test_color (str, optional): Hex color code for the test group plot.
                                    Defaults to "#FF7F0E".

    Returns:
        None: Displays the matplotlib plot.

    Raises:
        ValueError: If the input `data` is empty or lacks the 'scores' column.

    中文注释: 绘制匹配后测试组与对照组的分数分布
    """
    if data.empty:
        raise ValueError("No matched data found. Please run match() first.")
    if 'scores' not in data.columns:
        raise ValueError("No 'scores' column found in the matched dataset. Make sure scores are predicted.")

    sns.kdeplot(data[data[yvar] == 0]['scores'], label='Control (matched)', fill=True, color=control_color)
    sns.kdeplot(data[data[yvar] == 1]['scores'], label='Test (matched)', fill=True, color=test_color)
    plt.legend(loc='upper right')
    plt.xlim(0, 1)
    plt.title("Propensity Scores After Matching")
    plt.ylabel("Density")
    plt.xlabel("Scores")
    plt.show()

def plot_scores(data: pd.DataFrame, yvar: str, control_color: str = "#1F77B4", test_color: str = "#FF7F0E") -> None:
    """
    Plots the distribution of propensity scores before matching.

    Generates Kernel Density Estimate (KDE) plots to visualize the overlap
    of propensity scores between the test and control groups in the *original*
    (unmatched) dataset.

    Args:
        data (pd.DataFrame): The original DataFrame containing scores, must include
                             the `yvar` column and a 'scores' column.
        yvar (str): The name of the binary column indicating group membership (0 or 1).
        control_color (str, optional): Hex color code for the control group plot.
                                       Defaults to "#1F77B4".
        test_color (str, optional): Hex color code for the test group plot.
                                    Defaults to "#FF7F0E".

    Returns:
        None: Displays the matplotlib plot.

    Raises:
        ValueError: If the 'scores' column is not found in the input `data`.

    中文注释: 绘制匹配前测试组与对照组的分数分布
    """
    if 'scores' not in data.columns:
        raise ValueError("Propensity scores haven't been calculated. Please run predict_scores() first.")
    sns.kdeplot(data[data[yvar] == 0]['scores'], label='Control', fill=True, color=control_color)
    sns.kdeplot(data[data[yvar] == 1]['scores'], label='Test', fill=True, color=test_color)
    plt.legend(loc='upper right')
    plt.xlim(0, 1)
    plt.title("Propensity Scores Before Matching")
    plt.ylabel("Density")
    plt.xlabel("Scores")
    plt.show()


def compare_continuous(matcher, return_table: bool = False, plot_result: bool = True):
    """
    Compares continuous variables between groups before and after matching.

    For each continuous covariate identified in the `matcher` object:
    1. Calculates standardized median and mean differences before and after matching.
    2. Performs permutation tests based on Chi-square distance before and after matching.
    3. Performs bootstrap Kolmogorov-Smirnov (KS) tests before and after matching.
    4. If `plot_result` is True, generates side-by-side Empirical Cumulative
       Distribution Function (ECDF) plots comparing the distributions before and
       after matching, annotated with the calculated statistics (KS p-value,
       permutation p-value, standardized differences).
    5. Collects these statistics into a DataFrame.

    Args:
        matcher (Matcher): An instance of the `pysmatch.Matcher` class, which must
                           contain the original data (`matcher.data`), matched data
                           (`matcher.matched_data`), target variable name (`matcher.yvar`),
                           covariate list (`matcher.xvars`), and excluded columns list
                           (`matcher.exclude`).
        return_table (bool, optional): If True, returns the calculated statistics as a
                                       pandas DataFrame. Defaults to False.
        plot_result (bool, optional): If True, displays the ECDF comparison plots for
                                      each continuous variable. Defaults to True.

    Returns:
        Optional[pd.DataFrame]: If `return_table` is True, returns a DataFrame summarizing
                                the balance statistics for each continuous covariate.
                                Otherwise, returns None. Returns an empty DataFrame or None
                                if no continuous variables are found or if an error occurs.

    中文注释: 对连续变量在匹配前后做分布对比
    """
    test_results = []
    data = matcher.data
    matched_data = matcher.matched_data
    yvar = matcher.yvar

    for col in matched_data.columns:
        if uf.is_continuous(col, matcher.X) and col not in matcher.exclude:
            trb = matcher.test[col]
            cob = matcher.control[col]
            tra = matched_data[matched_data[yvar] == 1][col]
            coa = matched_data[matched_data[yvar] == 0][col]
            ecdf_trb = ECDF(trb)
            ecdf_cob = ECDF(cob)
            ecdf_tra = ECDF(tra)
            ecdf_coa = ECDF(coa)

            std_diff_med_before, std_diff_mean_before = uf.std_diff(trb.values, cob.values)
            std_diff_med_after, std_diff_mean_after = uf.std_diff(tra.values, coa.values)
            pb, _ = uf.grouped_permutation_test(uf.chi2_distance, trb.values, cob.values)
            pa, _ = uf.grouped_permutation_test(uf.chi2_distance, tra.values, coa.values)
            ksb = round(uf.ks_boot(trb.values, cob.values, nboots=1000), 6)
            ksa = round(uf.ks_boot(tra.values, coa.values, nboots=1000), 6)

            if plot_result:
                fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(12, 5))
                ax1.plot(ecdf_cob.x, ecdf_cob.y, label='Control', color=matcher.control_color)
                ax1.plot(ecdf_trb.x, ecdf_trb.y, label='Test', color=matcher.test_color)
                ax1.set_title(f"ECDF for {col} before Matching\n"
                              f"KS p-value: {ksb}\n"
                              f"Grouped Perm p-value: {pb}\n"
                              f"Std. Median Diff: {std_diff_med_before}\n"
                              f"Std. Mean Diff: {std_diff_mean_before}")
                ax2.plot(ecdf_coa.x, ecdf_coa.y, label='Control', color=matcher.control_color)
                ax2.plot(ecdf_tra.x, ecdf_tra.y, label='Test', color=matcher.test_color)
                ax2.set_title(f"ECDF for {col} after Matching\n"
                              f"KS p-value: {ksa}\n"
                              f"Grouped Perm p-value: {pa}\n"
                              f"Std. Median Diff: {std_diff_med_after}\n"
                              f"Std. Mean Diff: {std_diff_mean_after}")
                ax2.legend(loc="lower right")
                plt.xlim((0, np.percentile(ecdf_tra.x, 99)))
                plt.show()

            test_results.append({
                "var": col,
                "ks_before": ksb,
                "ks_after": ksa,
                "grouped_chisqr_before": pb,
                "grouped_chisqr_after": pa,
                "std_median_diff_before": std_diff_med_before,
                "std_median_diff_after": std_diff_med_after,
                "std_mean_diff_before": std_diff_mean_before,
                "std_mean_diff_after": std_diff_mean_after
            })

    if test_results:
        df_result = pd.DataFrame(test_results)[["var", "ks_before", "ks_after",
                                                "grouped_chisqr_before", "grouped_chisqr_after",
                                                "std_median_diff_before", "std_median_diff_after",
                                                "std_mean_diff_before", "std_mean_diff_after"]]
    else:
        df_result = pd.DataFrame()

    return df_result if return_table else None


def compare_categorical(matcher, return_table: bool = False, plot_result: bool = True):
    """
    Compares categorical variables between groups before and after matching.

    For each categorical covariate identified in the `matcher` object:
    1. Calculates the proportional difference (test % - control %) for each category
       level before and after matching.
    2. Performs Chi-Square tests of independence between the variable and the group
       indicator (`yvar`) before and after matching (using `matcher.prop_test`).
    3. If `plot_result` is True, generates bar plots showing the proportional
       differences for each category before and after matching, annotated with the
       Chi-Square p-values.
    4. Collects the Chi-Square test results into a DataFrame.

    Args:
        matcher (Matcher): An instance of the `pysmatch.Matcher` class, containing
                           original data, matched data, `yvar`, `xvars`, `exclude`.
        return_table (bool, optional): If True, returns the Chi-Square test results
                                       (variable name, p-value before, p-value after)
                                       as a pandas DataFrame. Defaults to False.
        plot_result (bool, optional): If True, displays the bar plots comparing
                                      proportional differences. Defaults to True.

    Returns:
        Optional[pd.DataFrame]: If `return_table` is True, returns a DataFrame summarizing
                                the Chi-Square test results for each categorical covariate.
                                Otherwise, returns None. Returns an empty DataFrame or None
                                if no categorical variables are found or if an error occurs.

    中文注释: 对分类变量在匹配前后做卡方检验并绘制比例差异图
    """
    data = matcher.data
    matched_data = matcher.matched_data
    yvar = matcher.yvar

    def prep_plot(df: pd.DataFrame, var: str, colname: str) -> pd.DataFrame:
        t = df[df[yvar] == 1]
        c = df[df[yvar] == 0]
        dummy = [col for col in t.columns if col not in {var, "match_id", "record_id", "weight"}][0]
        count_t = t[[var, dummy]].groupby(var).count() / len(t)
        count_c = c[[var, dummy]].groupby(var).count() / len(c)
        ret = (count_t - count_c).dropna()
        ret.columns = [colname]
        return ret

    test_results = []
    for col in matched_data.columns:
        if (not uf.is_continuous(col, matcher.X)) and (col not in matcher.exclude):
            dbefore = prep_plot(data, col, "before")
            dafter = prep_plot(matched_data, col, "after")
            df_plot = dbefore.join(dafter, how="outer").fillna(0)
            test_res = matcher.prop_test(col)
            if test_res is not None:
                test_results.append(test_res)
                if plot_result:
                    df_plot.plot.bar(alpha=0.8)
                    plt.title(f"Proportional Difference (test-control) for {col}\n"
                              f"Chi-Square p-value before: {test_res['before']} | after: {test_res['after']}")
                    ylim = max(0.09, abs(df_plot.values).max()) + 0.01
                    plt.ylim(-ylim, ylim)
                    plt.show()

    df_test_results = pd.DataFrame(test_results)[['var', 'before', 'after']] if test_results else pd.DataFrame()
    return df_test_results if return_table else None