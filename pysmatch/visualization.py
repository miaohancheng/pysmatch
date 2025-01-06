# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF

import pysmatch.utils as uf


def plot_scores(data, yvar, control_color="#1F77B4", test_color="#FF7F0E"):
    """
    Plots the distribution of propensity scores before matching between test and control.
    中文注释: 绘制匹配前测试组与对照组的分数分布
    """
    assert 'scores' in data.columns, \
        "Propensity scores haven't been calculated, please run predict_scores()"
    sns.kdeplot(data[data[yvar] == 0].scores, label='Control', fill=True, color=control_color)
    sns.kdeplot(data[data[yvar] == 1].scores, label='Test', fill=True, color=test_color)
    plt.legend(loc='upper right')
    plt.xlim((0, 1))
    plt.title("Propensity Scores Before Matching")
    plt.ylabel("Density")
    plt.xlabel("Scores")
    plt.show()


def compare_continuous(matcher, return_table: bool = False, plot_result: bool = True):
    """
    Plots the ECDFs for continuous features before and after matching.
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
            xtb, xcb = ECDF(trb), ECDF(cob)
            xta, xca = ECDF(tra), ECDF(coa)

            std_diff_med_before, std_diff_mean_before = uf.std_diff(trb, cob)
            std_diff_med_after, std_diff_mean_after = uf.std_diff(tra, coa)
            pb, truthb = uf.grouped_permutation_test(uf.chi2_distance, trb, cob)
            pa, trutha = uf.grouped_permutation_test(uf.chi2_distance, tra, coa)
            ksb = round(uf.ks_boot(trb, cob, nboots=1000), 6)
            ksa = round(uf.ks_boot(tra, coa, nboots=1000), 6)

            if plot_result:
                f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(12, 5))
                ax1.plot(xcb.x, xcb.y, label='Control', color=matcher.control_color)
                ax1.plot(xtb.x, xtb.y, label='Test', color=matcher.test_color)
                ax1.set_title(f'''
                    ECDF for {col} before Matching
                    KS p-value: {ksb}
                    Grouped Perm p-value: {pb}
                    Std. Median Difference: {std_diff_med_before}
                    Std. Mean Difference: {std_diff_mean_before}
                ''')
                ax2.plot(xca.x, xca.y, label='Control', color=matcher.control_color)
                ax2.plot(xta.x, xta.y, label='Test', color=matcher.test_color)
                ax2.set_title(f'''
                    ECDF for {col} after Matching
                    KS p-value: {ksa}
                    Grouped Perm p-value: {pa}
                    Std. Median Difference: {std_diff_med_after}
                    Std. Mean Difference: {std_diff_mean_after}
                ''')
                ax2.legend(loc="lower right")
                plt.xlim((0, np.percentile(xta.x, 99)))
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

    var_order = [
        "var",
        "ks_before",
        "ks_after",
        "grouped_chisqr_before",
        "grouped_chisqr_after",
        "std_median_diff_before",
        "std_median_diff_after",
        "std_mean_diff_before",
        "std_mean_diff_after"
    ]
    df_result = pd.DataFrame(test_results)[var_order] if test_results else pd.DataFrame()
    return df_result if return_table else None


def compare_categorical(matcher, return_table: bool = False, plot_result: bool = True):
    """
    Plots the proportional differences of each enumerated discrete column for test and control,
    and performs a Chi-Square Test of Independence before and after matching.
    中文注释: 对分类变量在匹配前后做卡方检验并绘制比例差异图
    """
    data = matcher.data
    matched_data = matcher.matched_data
    yvar = matcher.yvar

    def prep_plot(df: pd.DataFrame, var: str, colname: str) -> pd.DataFrame:
        t = df[df[yvar] == 1]
        c = df[df[yvar] == 0]
        dummy = [i for i in t.columns if i not in (var, "match_id", "record_id", "weight")][0]
        countt = t[[var, dummy]].groupby(var).count() / len(t)
        countc = c[[var, dummy]].groupby(var).count() / len(c)
        ret = (countt - countc).dropna()
        ret.columns = [colname]
        return ret

    title_str = '''
        Proportional Difference (test-control) for {} Before and After Matching
        Chi-Square Test for Independence p-value before | after:
        {} | {}
    '''
    test_results = []
    for col in matched_data.columns:
        if (not uf.is_continuous(col, matcher.X)) and (col not in matcher.exclude):
            dbefore = prep_plot(data, col, colname="before")
            dafter = prep_plot(matched_data, col, colname="after")
            df = dbefore.join(dafter, how="outer").fillna(0)
            test_results_i = matcher.prop_test(col)
            if test_results_i is not None:
                test_results.append(test_results_i)
                if plot_result:
                    df.plot.bar(alpha=.8)
                    plt.title(title_str.format(col, test_results_i["before"], test_results_i["after"]))
                    lim = max(.09, abs(df).max().max()) + .01
                    plt.ylim((-lim, lim))
                    plt.show()

    df_test_results = pd.DataFrame(test_results)[['var', 'before', 'after']] if test_results else pd.DataFrame()
    return df_test_results if return_table else None