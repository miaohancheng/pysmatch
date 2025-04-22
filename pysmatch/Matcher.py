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
    A class to perform propensity score matching (PSM).

    This class encapsulates the entire PSM workflow, including propensity
    score estimation, matching, and balance assessment.

    Attributes:
        data (pd.DataFrame): The input DataFrame containing treatment, outcome, and covariates.
        treatment (str): The name of the treatment column.
        outcome (str): The name of the outcome column.
        covariates (list): A list of covariate column names.
        exclude (list): A list of columns to exclude from calculations (often includes treatment and outcome).
        scores (pd.Series): Propensity scores estimated for each observation.
        matched_data (pd.DataFrame): DataFrame containing the matched pairs/groups.
        model_fit (object): The fitted propensity score model object.
        balance_stats (pd.DataFrame): Statistics assessing covariate balance before and after matching.
        n_matches (int): The number of matches to find for each treated unit (used in some matching methods).
        method (str): The matching method used (e.g., "nearest", "optimal", "radius").
    """
    def __init__(self, test: pd.DataFrame, control: pd.DataFrame, yvar: str,
                 formula: Optional[str] = None, exclude: Optional[List[str]] = None):
        """
        Initializes the Matcher object.

        Args:
            data (pd.DataFrame): The input DataFrame. Must contain treatment, outcome,
                and covariate columns.
            treatment (str): The name of the column indicating treatment status (e.g., 0 or 1).
            outcome (str): The name of the column containing the outcome variable.
            covariates (list): A list of column names to be used as covariates for
                propensity score estimation and matching.
            exclude (list, optional): A list of column names to exclude from internal
                calculations (typically includes treatment and outcome). If None, defaults
                to [treatment, outcome]. Defaults to None.

        Raises:
            ValueError: If input data types are incorrect or required columns are missing.
        """
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
        Fits a single propensity score model.

        Internal helper method that calls `pysmatch.modeling.fit_model`. This is
        typically used within `fit_scores`, especially when fitting multiple models
        for balancing or ensembling.

        Args:
            index (int): An identifier for the model (e.g., its index in an ensemble).
            X (pd.DataFrame): The feature matrix (covariates).
            y (pd.Series): The target variable (treatment indicator).
            model_type (str): The type of model to fit (e.g., 'linear', 'rf', 'gb').
            balance (bool): Whether the fitting process should aim to balance covariates
                            (e.g., by undersampling the majority class or using class weights).
            max_iter (int, optional): Maximum iterations for iterative solvers (like logistic
                                      regression). Defaults to 100.

        Returns:
            Dict[str, Any]: A dictionary containing the fitted model object under the key 'model'
                            and its accuracy under the key 'accuracy'.
        """
        return modeling.fit_model(index, X, y, model_type, balance, max_iter=max_iter)

    def fit_scores(self, balance: bool = True, nmodels: Optional[int] = None,
                   n_jobs: int = 1, model_type: str = 'linear',
                   max_iter: int = 100, use_optuna: bool = False,
                   n_trials: int = 10) -> None:
        """
        Fits propensity score model(s) to estimate scores.

        Supports single model fitting, ensemble fitting for balance (by undersampling
        the majority class across multiple models), or hyperparameter tuning using Optuna.

        Args:
            balance (bool, optional): If True, attempts to create balanced models. If `nmodels`
                                      is greater than 1, this typically involves fitting multiple
                                      models on undersampled majority data. If `nmodels` is 1,
                                      it might involve using class weights or other balancing
                                      techniques within the single model fit. Defaults to True.
            nmodels (Optional[int], optional): The number of models to fit in an ensemble.
                                               If None and `balance` is True, it's estimated based
                                               on the majority/minority class ratio. If None and
                                               `balance` is False, it defaults to 1. Ignored if
                                               `use_optuna` is True. Defaults to None.
            n_jobs (int, optional): The number of parallel jobs to run when fitting multiple
                                    models (`nmodels` > 1). Uses `ThreadPool`. Defaults to 1.
            model_type (str, optional): The type of classification model to use for propensity
                                        score estimation (e.g., 'linear' for Logistic Regression,
                                        'rf' for Random Forest, 'gb' for Gradient Boosting).
                                        Passed to `fit_model`. Defaults to 'linear'.
            max_iter (int, optional): Maximum iterations for the solver in iterative models like
                                      Logistic Regression. Passed to `fit_model`. Defaults to 100.
            use_optuna (bool, optional): If True, uses Optuna for hyperparameter tuning instead
                                         of fitting `nmodels`. `nmodels` is ignored. Defaults to False.
            n_trials (int, optional): The number of trials for Optuna optimization if `use_optuna`
                                      is True. Defaults to 10.

        Returns:
            None: Models and accuracies are stored in `self.models` and `self.model_accuracy`.
                  Propensity scores are calculated and stored later via `predict_scores()`.
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
        Predicts propensity scores using the fitted model(s).

        If multiple models were fitted (ensemble), the scores are averaged across models.
        The predicted scores are added to the `self.data` DataFrame as a 'scores' column.

        Returns:
            None: Scores are stored in `self.data['scores']`.

        Raises:
            RuntimeError: If `fit_scores()` has not been called successfully yet (no models exist).
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
        Performs matching based on estimated propensity scores.

        Args:
            method (str, optional): The matching algorithm to use.
                Options: "nearest", "optimal", "radius". Defaults to "nearest".
            n_matches (int, optional): The number of control units to match to each
                treated unit (for 'nearest' neighbor). Defaults to 1.
            caliper (float, optional): The maximum allowable distance (caliper) between
                propensity scores for a match. If None, no caliper is applied.
                Defaults to None. For 'radius' matching, this defines the radius.
            replace (bool, optional): Whether control units can be matched multiple times
                (matching with replacement). Defaults to False.
            **kwargs: Additional keyword arguments passed to the specific matching algorithm.

        Returns:
            pd.DataFrame: A DataFrame containing the matched treated and control units.

        Raises:
            RuntimeError: If propensity scores have not been estimated yet.
            ValueError: If an invalid matching method is specified.
        """
        self.matched_data = matching.perform_match(
            self.data, self.yvar, threshold=threshold,
            nmatches=nmatches, method=method, replacement=replacement
        )

    def plot_scores(self) -> None:
        """
        Plots the distribution of propensity scores before matching.

        Visualizes the overlap of scores between the test (treated) and control groups
        in the original (unmatched) data. Requires scores to be calculated first.
        """
        visualization.plot_scores(self.data, self.yvar,
                                  control_color=self.control_color,
                                  test_color=self.test_color)

    def tune_threshold(self, method: str, nmatches: int = 1,
                       rng: Optional[np.ndarray] = None) -> None:
        """
        Evaluates matching retention across a range of threshold values.

        Performs matching repeatedly for different threshold values and plots the
        proportion of the minority group retained at each threshold. This helps in
        selecting an appropriate threshold/caliper value.

        Args:
            method (str): The matching method to use (e.g., 'min', 'nn', 'radius') for
                          each threshold evaluation. Passed to `matching.tune_threshold`.
            nmatches (int, optional): The number of matches to seek (relevant for 'nn'/'min').
                                      Defaults to 1.
            rng (Optional[np.ndarray], optional): A NumPy array specifying the sequence of
                                                  threshold values to test. If None, a default
                                                  range (0 to 0.001 by 0.0001) is used.
                                                  Defaults to None.
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
        Calculates the frequency of each original record in the matched dataset.

        Useful when matching with replacement, as control units might appear multiple times.
        Requires `match()` to have been run successfully. The matched data must contain
        a 'match_id' or similar identifier linking back to original records if counts
        are desired per original record. *Correction: Based on `assign_weight_vector`,
        it seems 'record_id' is the key identifier.*

        Returns:
            pd.DataFrame: A DataFrame with columns like 'record_id' and 'n_records' (frequency count),
                          or an empty DataFrame if matching hasn't been done.
        """
        if self.matched_data.empty:
            logging.info("No matched data found. Please run match() first.")
            return pd.DataFrame()
        freqs = self.matched_data['match_id'].value_counts().reset_index()
        freqs.columns = ['freq', 'n_records']
        return freqs

    def assign_weight_vector(self) -> None:
        """
        Assigns inverse frequency weights to records in the matched dataset.

        Calculates weights as `1 / count`, where `count` is the number of times an
        original record (identified by `record_id`) appears in the matched dataset.
        This is often used in analyses after matching with replacement to account for
        controls matched multiple times. The weights are added as a 'weight' column
        to `self.matched_data`.

        Requires `match()` to have been run and `matched_data` to contain 'record_id'.
        """
        if self.matched_data.empty:
            logging.info("No matched data found. Please run match() first.")
            return
        record_freqs = self.matched_data.groupby('record_id').size().reset_index(name='count')
        record_freqs['weight'] = 1 / record_freqs['count']
        self.matched_data = self.matched_data.merge(record_freqs[['record_id', 'weight']], on='record_id')

    def prop_test(self, col: str) -> Optional[Dict[str, Any]]:
        """
        Performs Chi-Square tests for a categorical variable before and after matching.

        Compares the distribution of a categorical variable (`col`) between the test and
        control groups in both the original (`self.data`) and matched (`self.matched_data`)
        datasets using the Chi-Square test of independence.

        Args:
            col (str): The name of the categorical column to test. The method checks if the
                       column is likely categorical (not continuous) and not in `self.exclude`.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing the variable name ('var'), the
                                      p-value from the Chi-Square test before matching ('before'),
                                      and the p-value after matching ('after'). Returns None if
                                      the variable is continuous, excluded, or if tests fail.
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
        Prepares a contingency table for the Chi-Square test.

        Creates a cross-tabulation of the specified variable (`var`) against the
        treatment variable (`self.yvar`) from the given DataFrame. Handles potential
        missing categories by ensuring both treatment groups (0 and 1) are present
        as columns, filled with 0 counts if necessary.

        Args:
            data (pd.DataFrame): The DataFrame (either original or matched) to use.
            var (str): The categorical variable name.

        Returns:
            Optional[list]: A list-of-lists representation of the contingency table suitable
                            for `scipy.stats.chi2_contingency`. Returns None if the input
                            data is empty or the variable is missing.
        """
        counts = data.groupby([var, self.yvar]).size().unstack(fill_value=0)
        counts = counts.reindex(columns=[0, 1], fill_value=0)
        return counts.values.tolist()

    def compare_continuous(self, save: bool = False, return_table: bool = False, plot_result: bool = True):
        """
        Compares continuous variables between groups before and after matching.

        Delegates the comparison logic and plotting to `visualization.compare_continuous`.
        Typically calculates and displays standardized mean differences (SMD) or performs
        t-tests for all continuous covariates found in `self.xvars`.

        Args:
            save (bool, optional): Whether to save any generated plots (functionality depends
                                   on the implementation in `visualization.compare_continuous`).
                                   Defaults to False.
            return_table (bool, optional): If True, returns the comparison results as a
                                           DataFrame. Defaults to False.
            plot_result (bool, optional): If True, generates and displays plots (e.g., Love plot)
                                          summarizing the balance. Defaults to True.

        Returns:
            Optional[pd.DataFrame]: If `return_table` is True, returns a DataFrame containing
                                    the comparison statistics (e.g., SMD before/after).
                                    Otherwise, returns None.
        """
        return visualization.compare_continuous(self, return_table=return_table, plot_result=plot_result)

    def compare_categorical(self, return_table: bool = False, plot_result: bool = True):
        """
        Compares categorical variables between groups before and after matching.

        Delegates the comparison logic and plotting to `visualization.compare_categorical`.
        Typically calculates and displays differences in proportions or performs Chi-Square
        tests for all categorical covariates found in `self.xvars`.

        Args:
            return_table (bool, optional): If True, returns the comparison results as a
                                           DataFrame. Defaults to False.
            plot_result (bool, optional): If True, generates and displays plots summarizing
                                          the balance for categorical variables. Defaults to True.

        Returns:
            Optional[pd.DataFrame]: If `return_table` is True, returns a DataFrame containing
                                    the comparison statistics (e.g., p-values before/after).
                                    Otherwise, returns None.
        """
        return visualization.compare_categorical(self, return_table=return_table, plot_result=plot_result)

    def plot_matched_scores(self) -> None:
        """
        Plots the distribution of propensity scores after matching.

        Visualizes the score overlap between test and control groups specifically
        within the `self.matched_data`. Requires `match()` to have been run.
        """
        visualization.plot_matched_scores(
            self.matched_data,
            self.yvar,
            control_color=self.control_color,
            test_color=self.test_color
        )