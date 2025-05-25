# Matcher.py
# -*- coding: utf-8 -*-
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Any
from collections import defaultdict
from tqdm import tqdm

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
        data (pd.DataFrame): The input DataFrame containing treatment, outcome, covariates, and record_id.
        treatment_col (str): The name of the treatment column.
        yvar (str): The name of the treatment column (for compatibility with other modules).
        test (pd.DataFrame): The processed test/treated group DataFrame.
        control (pd.DataFrame): The processed control group DataFrame.
        exclude (list): A list of columns to exclude from calculations.
        scores (pd.Series): Propensity scores estimated for each observation, stored in `data['scores']`.
        matched_data (pd.DataFrame): DataFrame containing the matched pairs/groups.
        models (List[Any]): List of fitted propensity score model objects.
        model_accuracy (List[float]): List of accuracies for the fitted models.
        exhaustive_matching_default (bool): Default behavior for exhaustive matching for the instance.
    """
    def __init__(self, test: pd.DataFrame, control: pd.DataFrame, yvar: str,
                 formula: Optional[str] = None, exclude: Optional[List[str]] = None,
                 exhaustive_matching_default: bool = False):
        """
        Initializes the Matcher object.

        Args:
            test (pd.DataFrame): DataFrame for the test/treated group.
            control (pd.DataFrame): DataFrame for the control group.
            yvar (str): The name of the column indicating treatment status (e.g., 0 for control, 1 for test).
                        This column will be created in the combined data.
            formula (Optional[str]): R-style formula for propensity score model. If None,
                                     all columns not in `exclude` or `yvar` are used.
            exclude (Optional[List[str]]): A list of column names to exclude from
                                           covariates. `yvar` is automatically excluded.
            exhaustive_matching_default (bool, optional): Default setting for whether to use
                                                          exhaustive matching. Defaults to False.
        Raises:
            ValueError: If input data types are incorrect or required columns are missing.
        """
        if exclude is None:
            exclude = []

        plt.rcParams["figure.figsize"] = (10, 5)
        aux_match_cols = ['scores', 'match_id', 'weight', 'record_id', 'pair_score_diff', 'matched_as']

        t_df = test.copy()
        c_df = control.copy()

        t_df[yvar] = 1
        c_df[yvar] = 0

        t_df = t_df.dropna(axis=1, how="all")
        c_df = c_df.dropna(axis=1, how="all")

        self.data = pd.concat([t_df.reset_index(drop=True), c_df.reset_index(drop=True)], ignore_index=True)

        self.data['record_id'] = self.data.index

        self.control_color = "#1F77B4"
        self.test_color = "#FF7F0E"

        self.yvar = yvar
        self.treatment_col = yvar
        self.exclude = list(set(exclude + [self.treatment_col] + aux_match_cols))
        self.formula = formula
        self.nmodels = 1
        self.models: List[Any] = []
        self.swdata = None
        self.model_accuracy: List[float] = []
        self.errors = 0

        self.exhaustive_matching_default = exhaustive_matching_default

        self.data[self.treatment_col] = self.data[self.treatment_col].astype(int)

        potential_xvars = [col for col in self.data.columns if col not in self.exclude and col not in aux_match_cols]
        self.xvars = []
        for col in potential_xvars:
            if self.data[col].nunique(dropna=False) > 1:
                self.xvars.append(col)

        self.original_xvars = self.xvars.copy()

        if self.xvars:
            self.data = self.data.dropna(subset=self.xvars)

        if self.data.empty:
            logging.warning("DataFrame is empty after dropping NA values in covariates. Covariates may have too many NAs or data is too small.")
            self.X = pd.DataFrame(columns=self.xvars)
            self.y = pd.Series(dtype=int)
        else:
            self.X = self.data[self.xvars]
            self.y = self.data[self.treatment_col]

        self.matched_data = pd.DataFrame()

        self.test_df = self.data[self.data[self.treatment_col] == 1].copy()
        self.control_df = self.data[self.data[self.treatment_col] == 0].copy()

        self.test = self.test_df
        self.control = self.control_df

        self.testn = len(self.test_df)
        self.controln = len(self.control_df)

        if self.testn == 0 or self.controln == 0:
            logging.warning("No test or control samples after initial processing (e.g., NA drop). Matching may not be possible.")
            self.minority, self.majority = -1, -1
        elif self.testn <= self.controln:
            self.minority, self.majority = 1, 0
        else:
            self.minority, self.majority = 0, 1

        logging.info(f'Treatment column: {self.treatment_col}')
        logging.info(f'Covariates (xvars): {self.xvars}')
        if self.minority != -1:
            logging.info(f'N majority group (treatment={self.majority}): {len(self.data[self.data[self.treatment_col] == self.majority])}')
            logging.info(f'N minority group (treatment={self.minority}): {len(self.data[self.data[self.treatment_col] == self.minority])}')
        else:
            logging.info("Could not determine majority/minority due to empty or one-sided test/control groups.")


    def fit_model(self, index: int, X: pd.DataFrame, y: pd.Series, model_type: str,
                  balance: bool, max_iter: int = 100) -> Dict[str, Any]:
        """
        Fits a single propensity score model.

        Internal helper method that calls `pysmatch.modeling.fit_model`.
        Args:
            index (int): An identifier for the model.
            X (pd.DataFrame): The feature matrix (covariates).
            y (pd.Series): The target variable (treatment indicator).
            model_type (str): The type of model to fit (e.g., 'linear', 'rf', 'gb').
            balance (bool): Whether the fitting process should aim to balance covariates.
            max_iter (int, optional): Maximum iterations for iterative solvers. Defaults to 100.
        Returns:
            Dict[str, Any]: A dictionary containing the fitted model and its accuracy.
        """
        return modeling.fit_model(index, X, y, model_type, balance, max_iter=max_iter)

    def fit_scores(self, balance: bool = True, nmodels: Optional[int] = None,
                   n_jobs: int = 1, model_type: str = 'linear',
                   max_iter: int = 100, use_optuna: bool = False,
                   n_trials: int = 10) -> None:
        """
        Fits propensity score model(s) to estimate scores.

        Args:
            balance (bool, optional): If True, attempts to create balanced models. Defaults to True.
            nmodels (Optional[int], optional): Number of models for ensemble. Auto-estimated if None and balance=True.
            n_jobs (int, optional): Number of parallel jobs for ensemble fitting. Defaults to 1.
            model_type (str, optional): Type of model ('linear', 'rf', 'gb', 'knn', 'tree' for catboost). Defaults to 'linear'.
            max_iter (int, optional): Max iterations for solver. Defaults to 100.
            use_optuna (bool, optional): If True, use Optuna for hyperparameter tuning. Defaults to False.
            n_trials (int, optional): Number of Optuna trials if use_optuna is True. Defaults to 10.
        """
        from multiprocessing import cpu_count
        from multiprocessing.pool import ThreadPool

        self.models, self.model_accuracy = [], []
        self.model_type = model_type
        num_cores = cpu_count()
        n_jobs = min(num_cores, n_jobs if n_jobs > 0 else num_cores)
        logging.info(f"This computer has: {num_cores} cores, using {n_jobs} workers for fitting scores.")

        if self.X.empty or self.y.empty:
            logging.error("Feature matrix (X) or target vector (y) is empty (possibly due to NA dropping or small input). Cannot fit scores.")
            return

        if use_optuna:
            logging.info(f"Using Optuna for hyperparameter tuning with {n_trials} trials for model_type='{model_type}'.")
            result = modeling.optuna_tuner(self.X, self.y, model_type=model_type,
                                           n_trials=n_trials, balance=balance)
            self.models.append(result['model'])
            self.model_accuracy.append(result['accuracy'])
            logging.info(f"[Optuna] Best Accuracy: {result['accuracy']:.2%}")
            self.nmodels = 1
            return

        current_nmodels = nmodels
        if balance and current_nmodels is None:
            if self.minority != -1 and len(self.data[self.data[self.treatment_col] == self.minority]) > 0: # Check minority group is not empty
                minority_df_len = len(self.data[self.data[self.treatment_col] == self.minority])
                majority_df_len = len(self.data[self.data[self.treatment_col] == self.majority])
                if minority_df_len > 0:
                    ratio = majority_df_len / minority_df_len
                    current_nmodels = int(np.ceil(ratio))
                    current_nmodels = min(current_nmodels, 20)
                    current_nmodels = max(1, current_nmodels)
                else:
                    current_nmodels = 1
            else:
                current_nmodels = 1
        if current_nmodels is None:
            current_nmodels = 1

        self.nmodels = max(1, current_nmodels)

        logging.info(f"Fitting {self.nmodels} model(s) with balance={balance} for model_type='{model_type}'.")
        if balance and self.nmodels > 1:
            if len(self.data[self.data[self.treatment_col] == self.minority]) == 0 or \
                    len(self.data[self.data[self.treatment_col] == self.majority]) == 0:
                logging.warning(f"Cannot perform balanced ensemble fitting as one class has no samples. Fitting a single model instead.")
                # Fallback to fitting a single model
                result = self.fit_model(0, self.X, self.y, model_type, balance=False, max_iter=max_iter) # balance=False if one class is empty
                self.models.append(result['model'])
                self.model_accuracy.append(result['accuracy'])
                self.nmodels = 1
            else:
                with ThreadPool(n_jobs) as pool:
                    tasks = [
                        (i, self.X, self.y, model_type, balance, max_iter)
                        for i in range(self.nmodels)
                    ]
                    results = pool.starmap(self.fit_model, tasks)
                for res in results:
                    self.models.append(res['model'])
                    self.model_accuracy.append(res['accuracy'])
        else:
            result = self.fit_model(0, self.X, self.y, model_type, balance, max_iter)
            self.models.append(result['model'])
            self.model_accuracy.append(result['accuracy'])

        if self.model_accuracy:
            if self.nmodels > 1:
                avg_accuracy = np.mean(self.model_accuracy)
                logging.info(f"Average Accuracy over {self.nmodels} models: {avg_accuracy:.2%}")
            else:
                logging.info(f"Accuracy for single model: {self.model_accuracy[0]*100:.2f}%")
        else:
            logging.warning("No model accuracies recorded.")


    def predict_scores(self) -> None:
        """
        Predicts propensity scores using the fitted model(s).
        Scores are stored in `self.data['scores']`.
        """
        if not self.models:
            logging.warning("No trained models found. Please call fit_scores() first.")
            return
        if self.X.empty:
            logging.error("Feature matrix (X) is empty. Cannot predict scores.")
            return

        try:
            X_pred = self.X.copy()
            for col in X_pred.columns:
                if X_pred[col].dtype == 'object' or pd.api.types.is_string_dtype(X_pred[col]):
                    try:
                        X_pred[col] = pd.to_numeric(X_pred[col], errors='raise')
                    except ValueError:
                        # If conversion fails, log warning. Model might handle it or fail.
                        logging.warning(f"Column '{col}' is non-numeric and could not be auto-converted. "
                                        f"Model '{self.model_type}' might require numeric inputs or specific categorical handling.")

            model_preds = [model.predict_proba(X_pred)[:, 1] for model in self.models]
            scores = np.mean(model_preds, axis=0)
            self.data['scores'] = scores
            # Update self.test_df and self.control_df (and self.test, self.control) to include scores
            self.test_df['scores'] = self.data.loc[self.test_df.index, 'scores']
            self.control_df['scores'] = self.data.loc[self.control_df.index, 'scores']
            self.test = self.test_df.copy()
            self.control = self.control_df.copy()
            logging.info("Propensity scores predicted and added to 'scores' column in self.data, self.test_df, self.control_df.")
        except Exception as e:
            logging.error(f"Error during propensity score prediction: {e}")
            logging.error(f"Ensure covariates in self.X are numeric or appropriately preprocessed for model_type='{self.model_type}'.")


    def match(self, threshold: float = 0.001, nmatches: int = 1, method: str = 'min',
              replacement: bool = False, exhaustive_matching: Optional[bool] = None) -> None:
        """
        Performs matching based on estimated propensity scores.

        Args:
            threshold (float, optional): Threshold for score difference. Defaults to 0.001.
            nmatches (int, optional): Number of controls to match to each test unit. Defaults to 1.
            method (str, optional): The matching algorithm to use when exhaustive_matching is False.
                                    Passed to `pysmatch.matching.perform_match`. Defaults to 'min'.
            replacement (bool, optional): Whether controls can be matched to multiple test units
                                          when exhaustive_matching is False. Passed to `pysmatch.matching.perform_match`.
                                          Defaults to False.
            exhaustive_matching (Optional[bool], optional): If True, attempts to use a wider range of controls
                                                            by prioritizing unused or less-used controls.
                                                            If None, uses the instance's default
                                                            `self.exhaustive_matching_default`. Defaults to None.
        """
        if exhaustive_matching is None:
            exhaustive_matching = self.exhaustive_matching_default

        if 'scores' not in self.data.columns:
            logging.error("Propensity scores ('scores' column) not found in self.data. "
                          "Please run predict_scores() first.")
            self.matched_data = pd.DataFrame()
            return

        if 'record_id' not in self.data.columns:
            logging.error("'record_id' column not found in self.data. This is unexpected.")
            self.data['record_id'] = self.data.index # Fallback

        # Use self.test and self.control which should have scores if predict_scores was called
        current_test_df = self.test.copy() # self.test is self.test_df updated with scores
        current_control_df = self.control.copy() # self.control is self.control_df updated with scores


        if current_test_df.empty or current_control_df.empty:
            logging.warning("No test (treated) or control samples available for matching (self.test or self.control is empty).")
            self.matched_data = pd.DataFrame()
            return

        if 'scores' not in current_test_df.columns or 'scores' not in current_control_df.columns:
            logging.error("Propensity scores missing from internal test/control DataFrames. Ensure predict_scores() was effective.")
            self.matched_data = pd.DataFrame()
            return


        if exhaustive_matching:
            logging.info(f"Performing exhaustive matching: nmatches={nmatches}, threshold={threshold}")
            control_usage_counts = defaultdict(int)
            matched_pairs_info = []

            if 'record_id' not in current_test_df.columns or 'record_id' not in current_control_df.columns:
                logging.error("Record IDs missing from test_df or control_df used in exhaustive matching.")
                self.matched_data = pd.DataFrame()
                return

            for _, case_row in tqdm(current_test_df.iterrows(), total=len(current_test_df), desc="Exhaustive Matching"):
                case_prop_score = case_row['scores']
                case_record_id = case_row['record_id']

                temp_controls_df = current_control_df.copy()
                temp_controls_df['prop_score_diff'] = np.abs(temp_controls_df['scores'] - case_prop_score)

                eligible_controls_for_case = temp_controls_df[temp_controls_df['prop_score_diff'] <= threshold]

                if eligible_controls_for_case.empty:
                    continue

                eligible_controls_for_case['is_used'] = eligible_controls_for_case['record_id'].map(
                    lambda rid: control_usage_counts[rid] > 0
                )
                eligible_controls_for_case['usage_count'] = eligible_controls_for_case['record_id'].map(
                    lambda rid: control_usage_counts[rid]
                )

                eligible_controls_for_case = eligible_controls_for_case.sort_values(
                    by=['is_used', 'usage_count', 'prop_score_diff']
                )

                selected_controls_for_case = eligible_controls_for_case.head(nmatches)

                for _, control_row_selected in selected_controls_for_case.iterrows():
                    control_record_id = control_row_selected['record_id']
                    matched_pairs_info.append({
                        'case_record_id': case_record_id,
                        'control_record_id': control_record_id,
                        'score_diff': control_row_selected['prop_score_diff']
                    })
                    control_usage_counts[control_record_id] += 1

            if not matched_pairs_info:
                self.matched_data = pd.DataFrame()
                logging.info("No matches found with exhaustive matching.")
                return

            final_matched_rows_list = []
            current_match_id = 0
            for pair in matched_pairs_info:
                case_data_original_row = self.data[self.data['record_id'] == pair['case_record_id']]
                control_data_original_row = self.data[self.data['record_id'] == pair['control_record_id']]

                if case_data_original_row.empty or control_data_original_row.empty:
                    logging.warning(f"Could not find original data for pair: case_id {pair['case_record_id']}, control_id {pair['control_record_id']}. Skipping this pair.")
                    continue

                case_data_matched = case_data_original_row.iloc[0].copy()
                control_data_matched = control_data_original_row.iloc[0].copy()

                case_data_matched['match_id'] = current_match_id
                case_data_matched['matched_as'] = 'case'
                case_data_matched['pair_score_diff'] = pair['score_diff']

                control_data_matched['match_id'] = current_match_id
                control_data_matched['matched_as'] = 'control'
                control_data_matched['pair_score_diff'] = pair['score_diff']

                final_matched_rows_list.append(case_data_matched)
                final_matched_rows_list.append(control_data_matched)
                current_match_id += 1

            if final_matched_rows_list:
                self.matched_data = pd.DataFrame(final_matched_rows_list).reset_index(drop=True)
            else:
                self.matched_data = pd.DataFrame()
            logging.info(f"Exhaustive matching complete. {self.matched_data['match_id'].nunique() if not self.matched_data.empty else 0} pairs formed.")

        else:
            logging.info(f"Performing matching using pysmatch.matching.perform_match: method='{method}', replacement={replacement}, threshold={threshold}, nmatches={nmatches}")
            if 'scores' not in self.data.columns or 'record_id' not in self.data.columns or self.treatment_col not in self.data.columns:
                logging.error("self.data is missing required columns for standard matching. Aborting.")
                self.matched_data = pd.DataFrame()
                return

            self.matched_data = matching.perform_match(
                self.data,
                self.treatment_col,
                threshold=threshold,
                nmatches=nmatches,
                method=method,
                replacement=replacement
            )
            if self.matched_data.empty:
                logging.info("No matches found using pysmatch.matching.perform_match.")
            else:
                logging.info(f"Matching with pysmatch.matching.perform_match complete. Matched data has {len(self.matched_data)} rows.")
                if 'record_id' not in self.matched_data.columns:
                    logging.warning("'record_id' not found in matched_data from matching.perform_match. Subsequent weighting might fail.")
                if 'match_id' not in self.matched_data.columns:
                    logging.warning("'match_id' not found in matched_data from matching.perform_match. Record frequency might be affected.")


    def plot_scores(self) -> None:
        """
        Plots the distribution of propensity scores before matching.
        Visualizes score overlap between test (treated) and control groups.
        """
        if 'scores' not in self.data.columns:
            logging.warning("Scores not available in self.data. Cannot plot scores. Run predict_scores() first.")
            return
        visualization.plot_scores(self.data, self.treatment_col,
                                  control_color=self.control_color,
                                  test_color=self.test_color)

    def tune_threshold(self, method: str, nmatches: int = 1,
                       rng: Optional[np.ndarray] = None) -> None:
        """
        Evaluates matching retention across a range of threshold values.

        Args:
            method (str): Matching method ('min', 'nn', 'radius') for evaluation.
            nmatches (int, optional): Number of matches for 'nn'/'min'. Defaults to 1.
            rng (Optional[np.ndarray], optional): Threshold values to test. Defaults to np.arange(0, 0.001, 0.0001).
        """
        if 'scores' not in self.data.columns:
            logging.warning("Scores not available. Cannot tune threshold. Run predict_scores() first.")
            return
        if rng is None:
            rng = np.arange(0, 0.0011, 0.0001)

        thresholds, retained = matching.tune_threshold(self.data, self.treatment_col,
                                                       method=method, nmatches=nmatches, rng=rng)
        plt.figure(figsize=(10,6))
        plt.plot(thresholds, retained, marker='o')
        plt.title("Proportion of Minority Group Retained for Threshold Grid")
        plt.ylabel("Proportion Retained (Minority Group)")
        plt.xlabel("Threshold")
        plt.xticks(thresholds, rotation=45, ha="right")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def record_frequency(self) -> pd.DataFrame:
        """
        Calculates the frequency of each original control record in the matched dataset
        when exhaustive matching is used, or frequency of match_ids otherwise.
        Returns:
            pd.DataFrame: DataFrame with frequencies.
        """
        if self.matched_data.empty:
            logging.info("No matched data found. Cannot calculate record frequency.")
            return pd.DataFrame(columns=['record_id', 'n_occurrences_as_control'])

        if 'matched_as' in self.matched_data.columns and 'record_id' in self.matched_data.columns:
            control_records_in_matched = self.matched_data[self.matched_data['matched_as'] == 'control']
            if control_records_in_matched.empty:
                logging.info("No control records found in matched_data for frequency count.")
                return pd.DataFrame(columns=['record_id', 'n_occurrences_as_control'])
            freqs = control_records_in_matched['record_id'].value_counts().reset_index()
            freqs.columns = ['record_id', 'n_occurrences_as_control']
            return freqs
        elif 'match_id' in self.matched_data.columns:
            logging.warning("Calculating frequency based on 'match_id' as 'matched_as' column is not present.")
            freqs = self.matched_data['match_id'].value_counts().reset_index()
            freqs.columns = ['match_id', 'n_records_in_pair_id']
            return freqs
        else:
            logging.warning("Could not determine record frequency due to missing 'matched_as' or 'match_id' in matched_data.")
            return pd.DataFrame(columns=['id', 'frequency'])


    def assign_weight_vector(self) -> None:
        """
        Assigns inverse frequency weights to records in the matched dataset.
        Weights are 1/count, where count is how many times an original record
        (identified by `record_id`) appears. Added as 'weight' column.
        """
        if self.matched_data.empty:
            logging.info("No matched data found. Cannot assign weights.")
            return
        if 'record_id' not in self.matched_data.columns:
            logging.warning("'record_id' column not found in matched_data. Cannot assign weights.")
            return

        record_counts = self.matched_data['record_id'].value_counts()

        self.matched_data['weight'] = self.matched_data['record_id'].map(lambda rid: 1 / record_counts.get(rid, 1)) # Use .get for safety
        logging.info("Inverse frequency weights assigned to 'weight' column in matched_data.")


    def prop_test(self, col: str) -> Optional[Dict[str, Any]]:
        """
        Performs Chi-Square tests for a categorical variable before and after matching.

        Args:
            col (str): Name of the categorical column to test.
        Returns:
            Optional[Dict[str, Any]]: Dict with 'var', 'before' p-value, 'after' p-value. None if error.
        """
        from scipy import stats
        if col not in self.data.columns or (not self.matched_data.empty and col not in self.matched_data.columns):
            logging.warning(f"Column '{col}' not found in original or matched data for prop_test.")
            return None
        if col == self.treatment_col or col in self.exclude:
            logging.info(f"Column '{col}' is the treatment variable or in the global exclude list. Skipping prop_test.")
            return None

        is_col_continuous = False
        if col in self.data.columns:
            is_col_continuous = uf.is_continuous(col, self.data[[col]]) if hasattr(uf, 'is_continuous') else self.data[col].dtype.kind in 'ifc'

        if not is_col_continuous:
            try:
                before_data_table = self.prep_prop_test(self.data, col)
                pval_before = round(stats.chi2_contingency(before_data_table)[1], 6) if before_data_table is not None and len(before_data_table) > 0 and np.array(before_data_table).size > 0 else np.nan
            except ValueError as e:
                logging.warning(f"Chi-square test failed for '{col}' before matching (possibly all values are same in a group or other data issue): {e}")
                pval_before = np.nan

            pval_after = np.nan
            if not self.matched_data.empty:
                try:
                    after_data_table = self.prep_prop_test(self.matched_data, col)
                    pval_after = round(stats.chi2_contingency(after_data_table)[1], 6) if after_data_table is not None and len(after_data_table) > 0 and np.array(after_data_table).size > 0 else np.nan
                except ValueError as e:
                    logging.warning(f"Chi-square test failed for '{col}' after matching: {e}")
                    pval_after = np.nan

            return {'var': col, 'before': pval_before, 'after': pval_after}
        else:
            logging.info(f"{col} is detected as a continuous variable. Skipping prop_test.")
            return None

    def prep_prop_test(self, data: pd.DataFrame, var: str) -> Optional[list]:
        """
        Prepares a contingency table for the Chi-Square test.

        Args:
            data (pd.DataFrame): The DataFrame (original or matched) to use.
            var (str): The categorical variable name.
        Returns:
            Optional[list]: List-of-lists for `scipy.stats.chi2_contingency`. None if error.
        """
        if data.empty or var not in data.columns or self.treatment_col not in data.columns:
            logging.warning(f"Cannot prep_prop_test for '{var}': data is empty or columns missing.")
            return None
        try:
            if data[var].nunique(dropna=False) < 1 or data[self.treatment_col].nunique(dropna=False) < 1 : # Should be < 2 for at least two categories/groups
                logging.info(f"Not enough distinct values for '{var}' or '{self.treatment_col}' in provided data for chi-square test prep.")
                return None

            # Ensure at least two categories in var and two groups in treatment_col for a valid chi-square
            if data[var].nunique() < 2 or data[self.treatment_col].nunique() < 2:
                logging.info(f"Chi-square test requires at least two categories for '{var}' and two groups for '{self.treatment_col}'.")
                return None


            counts = pd.crosstab(data[var], data[self.treatment_col])

            if 0 not in counts.columns: counts[0] = 0
            if 1 not in counts.columns: counts[1] = 0
            counts = counts[[0, 1]]

            if (counts.sum().sum() == 0) or (counts.shape[0] < 2) or (counts.shape[1] < 2) : # Check for empty or degenerate table
                logging.info(f"Contingency table for '{var}' is degenerate (e.g., all zeros, or not 2x2). Skipping chi-square.")
                return None

            if (counts.values < 0).any():
                logging.warning(f"Contingency table for '{var}' contains negative values. This is unexpected.")
                return None

            return counts.values.tolist()
        except Exception as e:
            logging.error(f"Error preparing contingency table for '{var}': {e}")
            return None


    def compare_continuous(self, save: bool = False, return_table: bool = False, plot_result: bool = True):
        """
        Compares continuous variables between groups before and after matching.
        Delegates to `visualization.compare_continuous`.
        The `visualization` module is expected to use `self.yvar` (which is `self.treatment_col`).
        """
        if not hasattr(visualization, 'compare_continuous'):
            logging.error("visualization.compare_continuous function not found.")
            return None if return_table else None
        return visualization.compare_continuous(self, return_table=return_table, plot_result=plot_result)

    def compare_categorical(self, return_table: bool = False, plot_result: bool = True):
        """
        Compares categorical variables between groups before and after matching.
        Delegates to `visualization.compare_categorical`.
        The `visualization` module is expected to use `self.yvar` (which is `self.treatment_col`).
        """
        if not hasattr(visualization, 'compare_categorical'):
            logging.error("visualization.compare_categorical function not found.")
            return None if return_table else None
        return visualization.compare_categorical(self, return_table=return_table, plot_result=plot_result)

    def plot_matched_scores(self) -> None:
        """
        Plots the distribution of propensity scores after matching.
        Visualizes score overlap in `self.matched_data`.
        """
        if self.matched_data.empty:
            logging.warning("Matched data is empty. Cannot plot matched scores.")
            return
        if 'scores' not in self.matched_data.columns:
            logging.warning("Scores not available in self.matched_data. Cannot plot matched scores.")
            return
        if self.treatment_col not in self.matched_data.columns:
            logging.warning(f"Treatment column '{self.treatment_col}' not in self.matched_data. Cannot plot matched scores.")
            return

        visualization.plot_matched_scores(
            self.matched_data,
            self.treatment_col,
            control_color=self.control_color,
            test_color=self.test_color
        )
