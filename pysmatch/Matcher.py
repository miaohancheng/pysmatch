# -*- coding: utf-8 -*-
from __future__ import print_function
import logging
import numpy as np
import pandas as pd
import patsy
import matplotlib.pyplot as plt

from typing import List, Optional

import pysmatch.utils as uf
from pysmatch.modeling import fit_model, optuna_tuner


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class Matcher:
    """
    Matcher Class -- Match data for an observational study.
    中文注释: 匹配器类, 用于观察性研究中进行配对
    """

    def __init__(self, test: pd.DataFrame, control: pd.DataFrame, yvar: str,
                 formula: Optional[str] = None, exclude: Optional[List[str]] = None):
        if exclude is None:
            exclude = []
        plt.rcParams["figure.figsize"] = (10, 5)
        aux_match = ['scores', 'match_id', 'weight', 'record_id']
        t = test.copy().reset_index(drop=True)
        c = control.copy().reset_index(drop=True)
        t = t.dropna(axis=1, how="all")
        c = c.dropna(axis=1, how="all")
        c.index += len(t)
        self.data = pd.concat([t, c], ignore_index=True)
        self.control_color = "#1F77B4"
        self.test_color = "#FF7F0E"
        self.yvar = yvar
        self.exclude = exclude + [self.yvar] + aux_match
        self.formula = formula
        self.nmodels = 1
        self.models = []
        self.swdata = None
        self.model_accuracy = []
        self.errors = 0
        self.data[yvar] = self.data[yvar].astype(int)  # should be binary 0, 1
        self.xvars = [i for i in self.data.columns if i not in self.exclude]
        self.original_xvars = self.xvars.copy()
        self.data = self.data.dropna(subset=self.xvars)
        self.matched_data = []
        self.xvars_escaped = [f"Q('{x}')" for x in self.xvars]
        self.yvar_escaped = f"Q('{self.yvar}')"

        # design matrices
        self.y, self.X = patsy.dmatrices(
            f'{self.yvar_escaped} ~ {" + ".join(self.xvars_escaped)}',
            data=self.data, return_type='dataframe'
        )
        self.design_info = self.X.design_info
        self.test = self.data[self.data[yvar] == 1]
        self.control = self.data[self.data[yvar] == 0]
        self.testn = len(self.test)
        self.controln = len(self.control)

        if self.testn <= self.controln:
            self.minority, self.majority = 1, 0
        else:
            self.minority, self.majority = 0, 1

        logging.info(f'Formula: {yvar} ~ {"+".join(self.xvars)}')
        logging.info(f'n majority: {len(self.data[self.data[yvar] == self.majority])}')
        logging.info(f'n minority: {len(self.data[self.data[yvar] == self.minority])}')


    def fit_model(self, index: int, X: pd.DataFrame, y: pd.Series, model_type: str,
                  balance: bool, max_iter: int = 100) -> dict:
        """
        Internal helper that calls modeling.fit_model
        """
        return fit_model(index, X, y, model_type, balance, max_iter=max_iter)


    def fit_scores(self, balance: bool = True, nmodels: Optional[int] = None,
                   n_jobs: int = 1, model_type: str = 'linear',
                   max_iter: int = 100, use_optuna: bool = False,
                   n_trials: int = 10):
        """
        Fits one or multiple models to get propensity scores.
        中文注释: 训练(多个)模型以获取倾向分数
        新增: 如果 use_optuna=True, 则使用 optuna 搜索超参(只训练1个最佳模型)
        """
        import multiprocessing as mp
        from multiprocessing.pool import ThreadPool as Pool

        self.models, self.model_accuracy = [], []
        self.model_type = model_type
        num_cores = mp.cpu_count()
        logging.info(f"This computer has: {num_cores} cores, The workers will be: {min(num_cores, n_jobs)}")

        if use_optuna:
            result = optuna_tuner(self.X, self.y, model_type=model_type,
                                  n_trials=n_trials, balance=balance)
            self.models.append(result['model'])
            self.model_accuracy.append(result['accuracy'])
            logging.info(f"Optuna Best Accuracy: {result['accuracy']:.2%}")
            return

        if balance and nmodels is None:
            minor, major = [self.data[self.data[self.yvar] == i] for i in (self.minority, self.majority)]
            import numpy as np
            nmodels = int(np.ceil((len(major) / len(minor)) / 10) * 10)
        if nmodels is None:
            nmodels = 1

        nmodels = max(1, nmodels)

        if balance and nmodels > 1:
            with Pool(min(num_cores, n_jobs)) as pool:
                tasks = [
                    (i, self.X, self.y, model_type, balance, max_iter)
                    for i in range(nmodels)
                ]
                results = pool.starmap(self.fit_model, tasks)
            for res in results:
                self.models.append(res['model'])
                self.model_accuracy.append(res['accuracy'])
            import numpy as np
            logging.info(f"Average Accuracy: {np.mean(self.model_accuracy):.2%}")
        else:
            result = self.fit_model(0, self.X, self.y, model_type, balance, max_iter)
            self.models.append(result['model'])
            self.model_accuracy.append(result['accuracy'])
            logging.info(f"Accuracy: {self.model_accuracy[0]*100:.2f}%")

    def predict_scores(self):
        """
        Predict propensity scores using the trained models.
        中文注释: 使用已训练的模型预测倾向分数
        """
        if not self.models:
            logging.warning("No trained models found. Call fit_scores() first.")
            return
        model_preds = []
        for m in self.models:
            preds = m.predict_proba(self.X)[:, 1]
            model_preds.append(preds)
        model_preds = np.array(model_preds)
        scores = np.mean(model_preds, axis=0)
        self.data['scores'] = scores

    def match(self, threshold: float = 0.001, nmatches: int = 1, method: str = 'min',
              replacement: bool = False):
        """
        Finds suitable match(es) for each record in the minority dataset.
        中文注释: 调用匹配方法，对少数类样本找到匹配记录
        """
        from pysmatch.matching import perform_match
        matched_df = perform_match(
            self.data, self.yvar, threshold=threshold,
            nmatches=nmatches, method=method, replacement=replacement
        )
        self.matched_data = matched_df

    def plot_scores(self):
        """
        Plots the distribution of propensity scores before matching.
        中文注释: 绘制匹配前的倾向分数分布
        """
        from pysmatch.visualization import plot_scores
        plot_scores(self.data, self.yvar, control_color=self.control_color, test_color=self.test_color)

    def tune_threshold(self, method: str, nmatches: int = 1,
                       rng: np.ndarray = np.arange(0, .001, .0001)):
        """
        Matches data over a grid to optimize threshold value and plots results.
        中文注释: 在一系列阈值范围上做匹配并绘制保留率
        """
        from pysmatch.matching import tune_threshold
        thresholds, retained = tune_threshold(self.data, self.yvar,
                                              method=method, nmatches=nmatches, rng=rng)
        plt.plot(thresholds, retained)
        plt.title("Proportion of Data retained for grid of threshold values")
        plt.ylabel("Proportion Retained")
        plt.xlabel("Threshold")
        plt.xticks(thresholds, rotation=90)
        plt.show()

    def record_frequency(self) -> pd.DataFrame:
        """
        Calculates the frequency of specific records in the matched dataset.
        中文注释: 计算匹配后数据中记录出现的次数
        """
        if len(self.matched_data) == 0:
            logging.info("No matched data found. Please run match() first.")
            return pd.DataFrame()
        freqs = self.matched_data['match_id'].value_counts().reset_index()
        freqs.columns = ['freq', 'n_records']
        return freqs

    def assign_weight_vector(self):
        """
        Assigns an inverse frequency weight to each record in the matched dataset.
        中文注释: 为匹配后的每条记录分配一个“1 / 匹配次数”的权重
        """
        if len(self.matched_data) == 0:
            logging.info("No matched data found. Please run match() first.")
            return
        record_freqs = self.matched_data.groupby('record_id').size().reset_index(name='count')
        record_freqs['weight'] = 1 / record_freqs['count']
        self.matched_data = self.matched_data.merge(record_freqs[['record_id', 'weight']], on='record_id')

    def prop_test(self, col: str) -> Optional[dict]:
        """
        Performs a Chi-Square test of independence on <col>
        中文注释: 对某个离散变量做卡方检验
        """
        from scipy import stats
        if not uf.is_continuous(col, self.X) and col not in self.exclude:
            pval_before = round(stats.chi2_contingency(self.prep_prop_test(self.data, col))[1], 6)
            pval_after = round(stats.chi2_contingency(self.prep_prop_test(self.matched_data, col))[1], 6)
            return {'var': col, 'before': pval_before, 'after': pval_after}
        else:
            logging.info(f"{col} is a continuous variable or excluded.")
            return None

    def prep_prop_test(self, data: pd.DataFrame, var: str):
        """
        Helper method for running chi-square contingency tests.
        中文注释: 卡方检验辅助方法，补全空的类别计数
        """
        counts = data.groupby([var, self.yvar]).size().unstack(fill_value=0)
        if 0 not in counts.columns:
            counts[0] = 0
        if 1 not in counts.columns:
            counts[1] = 0
        counts = counts[[0, 1]]
        return counts.values.tolist()

    def compare_continuous(self, save: bool = False, return_table: bool = False, plot_result: bool = True):
        """
        Wrapper to call visualization.compare_continuous
        """
        from pysmatch.visualization import compare_continuous
        return compare_continuous(self, return_table=return_table, plot_result=plot_result)

    def compare_categorical(self, return_table: bool = False, plot_result: bool = True):
        """
        Wrapper to call visualization.compare_categorical
        """
        from pysmatch.visualization import compare_categorical
        return compare_categorical(self, return_table=return_table, plot_result=plot_result)