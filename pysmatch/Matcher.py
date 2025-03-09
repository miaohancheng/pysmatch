# Matcher.py
# -*- coding: utf-8 -*-
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Any

from pysmatch import utils as uf
from pysmatch import modeling
from pysmatch import matching
from pysmatch import visualization

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
        if yvar not in test.columns or yvar not in control.columns:
            raise ValueError(f"'{yvar}' must be present in both test and control DataFrames.")

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
        self.models: List[Any] = []
        self.swdata = None
        self.model_accuracy: List[float] = []
        self.errors = 0

        # 确保yvar为二元变量
        self.data[self.yvar] = self.data[self.yvar].astype(int)
        self.xvars = [col for col in self.data.columns if col not in self.exclude]
        self.original_xvars = self.xvars.copy()
        self.data = self.data.dropna(subset=self.xvars)
        self.matched_data = pd.DataFrame()
        self.X = self.data[self.xvars]
        self.y = self.data[self.yvar]
        self.test = self.data[self.data[self.yvar] == 1]
        self.control = self.data[self.data[self.yvar] == 0]
        self.testn = len(self.test)
        self.controln = len(self.control)
        if self.testn <= self.controln:
            self.minority, self.majority = 1, 0
        else:
            self.minority, self.majority = 0, 1
        logging.info(f'Formula: {yvar} ~ {" + ".join(self.xvars)}')
        logging.info(f'n majority: {len(self.data[self.data[self.yvar] == self.majority])}')
        logging.info(f'n minority: {len(self.data[self.data[self.yvar] == self.minority])}')

    def fit_model(self, index: int, X: pd.DataFrame, y: pd.Series, model_type: str,
                  balance: bool, max_iter: int = 100) -> Dict[str, Any]:
        """
        Internal helper that calls modeling.fit_model.
        """
        return modeling.fit_model(index, X, y, model_type, balance, max_iter=max_iter)

    def fit_scores(self, balance: bool = True, nmodels: Optional[int] = None,
                   n_jobs: int = 1, model_type: str = 'linear',
                   max_iter: int = 100, use_optuna: bool = False,
                   n_trials: int = 10) -> None:
        """
        Fits one or multiple models to get propensity scores.
        中文注释: 训练(多个)模型以获取倾向分数
        """
        from multiprocessing import cpu_count
        from multiprocessing.pool import ThreadPool

        self.models, self.model_accuracy = [], []
        self.model_type = model_type
        num_cores = cpu_count()
        n_jobs = min(num_cores, n_jobs)
        logging.info(f"This computer has: {num_cores} cores, using {n_jobs} workers.")

        if use_optuna:
            result = modeling.optuna_tuner(self.X, self.y, model_type=model_type,
                                           n_trials=n_trials, balance=balance)
            self.models.append(result['model'])
            self.model_accuracy.append(result['accuracy'])
            logging.info(f"[Optuna] Best Accuracy: {result['accuracy']:.2%}")
            return

        if balance and nmodels is None:
            # 根据少数/多数样本比例估计模型数，用于集成
            minority_df = self.data[self.data[self.yvar] == self.minority]
            majority_df = self.data[self.data[self.yvar] == self.majority]
            nmodels = int(np.ceil((len(majority_df) / len(minority_df)) / 10) * 10)
        if nmodels is None:
            nmodels = 1
        nmodels = max(1, nmodels)

        if balance and nmodels > 1:
            with ThreadPool(n_jobs) as pool:
                tasks = [
                    (i, self.X, self.y, model_type, balance, max_iter)
                    for i in range(nmodels)
                ]
                results = pool.starmap(self.fit_model, tasks)
            for res in results:
                self.models.append(res['model'])
                self.model_accuracy.append(res['accuracy'])
            avg_accuracy = np.mean(self.model_accuracy)
            logging.info(f"Average Accuracy: {avg_accuracy:.2%}")
        else:
            result = self.fit_model(0, self.X, self.y, model_type, balance, max_iter)
            self.models.append(result['model'])
            self.model_accuracy.append(result['accuracy'])
            logging.info(f"Accuracy: {self.model_accuracy[0]*100:.2f}%")

    def predict_scores(self) -> None:
        """
        Predict propensity scores using the trained models.
        中文注释: 使用已训练的模型预测倾向分数
        """
        if not self.models:
            logging.warning("No trained models found. Please call fit_scores() first.")
            return
        model_preds = [model.predict_proba(self.X)[:, 1] for model in self.models]
        scores = np.mean(model_preds, axis=0)
        self.data['scores'] = scores

    def match(self, threshold: float = 0.001, nmatches: int = 1, method: str = 'min',
              replacement: bool = False) -> None:
        """
        Finds suitable match(es) for each record in the minority dataset.
        中文注释: 调用匹配方法，对少数类样本找到匹配记录
        """
        self.matched_data = matching.perform_match(
            self.data, self.yvar, threshold=threshold,
            nmatches=nmatches, method=method, replacement=replacement
        )

    def plot_scores(self) -> None:
        """
        Plots the distribution of propensity scores before matching.
        中文注释: 绘制匹配前的倾向分数分布
        """
        visualization.plot_scores(self.data, self.yvar,
                                  control_color=self.control_color,
                                  test_color=self.test_color)

    def tune_threshold(self, method: str, nmatches: int = 1,
                       rng: Optional[np.ndarray] = None) -> None:
        """
        Matches data over a grid to optimize threshold value and plots results.
        中文注释: 在一系列阈值范围上做匹配并绘制保留率
        """
        if rng is None:
            rng = np.arange(0, 0.001, 0.0001)
        thresholds, retained = matching.tune_threshold(self.data, self.yvar,
                                                       method=method, nmatches=nmatches, rng=rng)
        plt.plot(thresholds, retained)
        plt.title("Proportion of Data Retained for Threshold Grid")
        plt.ylabel("Proportion Retained")
        plt.xlabel("Threshold")
        plt.xticks(thresholds, rotation=90)
        plt.show()

    def record_frequency(self) -> pd.DataFrame:
        """
        Calculates the frequency of specific records in the matched dataset.
        中文注释: 计算匹配后数据中记录出现的次数
        """
        if self.matched_data.empty:
            logging.info("No matched data found. Please run match() first.")
            return pd.DataFrame()
        freqs = self.matched_data['match_id'].value_counts().reset_index()
        freqs.columns = ['freq', 'n_records']
        return freqs

    def assign_weight_vector(self) -> None:
        """
        Assigns an inverse frequency weight to each record in the matched dataset.
        中文注释: 为匹配后的每条记录分配一个“1 / 匹配次数”的权重
        """
        if self.matched_data.empty:
            logging.info("No matched data found. Please run match() first.")
            return
        record_freqs = self.matched_data.groupby('record_id').size().reset_index(name='count')
        record_freqs['weight'] = 1 / record_freqs['count']
        self.matched_data = self.matched_data.merge(record_freqs[['record_id', 'weight']], on='record_id')

    def prop_test(self, col: str) -> Optional[Dict[str, Any]]:
        """
        Performs a Chi-Square test of independence on <col>
        中文注释: 对某个离散变量做卡方检验
        """
        from scipy import stats
        if not uf.is_continuous(col, self.X) and col not in self.exclude:
            before_data = self.prep_prop_test(self.data, col)
            after_data = self.prep_prop_test(self.matched_data, col)
            pval_before = round(stats.chi2_contingency(before_data)[1], 6)
            pval_after = round(stats.chi2_contingency(after_data)[1], 6)
            return {'var': col, 'before': pval_before, 'after': pval_after}
        else:
            logging.info(f"{col} is a continuous variable or excluded.")
            return None

    def prep_prop_test(self, data: pd.DataFrame, var: str) -> list:
        """
        Helper method for running chi-square contingency tests.
        中文注释: 卡方检验辅助方法，补全空的类别计数
        """
        counts = data.groupby([var, self.yvar]).size().unstack(fill_value=0)
        counts = counts.reindex(columns=[0, 1], fill_value=0)
        return counts.values.tolist()

    def compare_continuous(self, save: bool = False, return_table: bool = False, plot_result: bool = True):
        """
        Wrapper to call visualization.compare_continuous
        """
        return visualization.compare_continuous(self, return_table=return_table, plot_result=plot_result)

    def compare_categorical(self, return_table: bool = False, plot_result: bool = True):
        """
        Wrapper to call visualization.compare_categorical
        """
        return visualization.compare_categorical(self, return_table=return_table, plot_result=plot_result)

    def plot_matched_scores(self) -> None:
        """
        Plots the distribution of propensity scores after matching.
        中文注释: 绘制匹配后倾向分数分布
        """
        visualization.plot_matched_scores(
            self.matched_data,
            self.yvar,
            control_color=self.control_color,
            test_color=self.test_color
        )