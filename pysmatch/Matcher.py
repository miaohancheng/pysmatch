# Matcher.py
# -*- coding: utf-8 -*-
import logging
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any

from pysmatch import utils as uf
from pysmatch import modeling
from pysmatch import matching


def _visualization_module():
    """Import optional plotting dependencies with an actionable message."""

    try:
        from pysmatch import visualization
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Plotting and balance visualizations require optional dependencies; "
            "install them with `pip install 'pysmatch[viz]'`."
        ) from exc
    return visualization


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
                 exhaustive_matching_default: bool = False,
                 random_state: int = 42):
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
            random_state (int, optional): Base seed used by model fitting, threshold tuning,
                                         and random matching. Defaults to 42.
        Raises:
            ValueError: If input data types are incorrect or required columns are missing.
        """
        if exclude is None:
            exclude = []
        if random_state is None:
            random_state = 42
        if not isinstance(random_state, (int, np.integer)):
            raise TypeError("random_state must be an integer or None.")

        aux_match_cols = ['scores', 'match_id', 'weight', 'record_id', 'pair_score_diff', 'matched_as']

        t_df = test.copy()
        c_df = control.copy()

        t_df[yvar] = 1
        c_df[yvar] = 0

        t_df = t_df.dropna(axis=1, how="all")
        c_df = c_df.dropna(axis=1, how="all")

        self.data = pd.concat([t_df.reset_index(drop=True), c_df.reset_index(drop=True)], ignore_index=True)
        if 'record_id' in self.data.columns and self.data['record_id'].notna().all() and self.data['record_id'].is_unique:
            self.record_id_source = "input data"
        else:
            if 'record_id' in self.data.columns:
                logging.warning(
                    "Input record_id column has missing or duplicate values. Falling back to generated row-based record_id values."
                )
            self.data['record_id'] = self.data.index
            self.record_id_source = "generated index"

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
        self.model_metrics: List[Dict[str, float]] = []
        self.validation_y: Optional[np.ndarray] = None
        self.validation_scores: Optional[np.ndarray] = None
        self.errors = 0

        self.exhaustive_matching_default = exhaustive_matching_default
        self.random_state = random_state

        self.data[self.treatment_col] = self.data[self.treatment_col].astype(int)

        raw_row_count = len(self.data)
        raw_test_count = int((self.data[self.treatment_col] == 1).sum())
        raw_control_count = int((self.data[self.treatment_col] == 0).sum())

        self.xvars, self.data, design_matrix = self._prepare_design_data(
            data=self.data,
            aux_match_cols=aux_match_cols,
        )
        self.original_xvars = self.xvars.copy()
        self.rows_dropped_due_to_na = raw_row_count - len(self.data)
        self.test_rows_dropped_due_to_na = raw_test_count - int((self.data[self.treatment_col] == 1).sum())
        self.control_rows_dropped_due_to_na = raw_control_count - int((self.data[self.treatment_col] == 0).sum())

        if self.rows_dropped_due_to_na > 0:
            logging.warning(
                "Dropped %s row(s) with missing covariates%s. Test rows dropped: %s. Control rows dropped: %s.",
                self.rows_dropped_due_to_na,
                f" using formula '{self.formula}'" if self.formula else "",
                self.test_rows_dropped_due_to_na,
                self.control_rows_dropped_due_to_na,
            )

        if self.data.empty:
            logging.warning("DataFrame is empty after dropping NA values in covariates. Covariates may have too many NAs or data is too small.")
            self.X = pd.DataFrame(columns=self.xvars)
            self.y = pd.Series(dtype=int)
        else:
            self.X = design_matrix
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
        if self.formula:
            logging.info(f'Formula: {self.formula}')
        logging.info(f'Covariates (xvars): {self.xvars}')
        logging.info(f"Record ID source: {self.record_id_source}")
        if self.minority != -1:
            logging.info(f'N majority group (treatment={self.majority}): {len(self.data[self.data[self.treatment_col] == self.majority])}')
            logging.info(f'N minority group (treatment={self.minority}): {len(self.data[self.data[self.treatment_col] == self.minority])}')
        else:
            logging.info("Could not determine majority/minority due to empty or one-sided test/control groups.")

    def _prepare_design_data(
        self,
        data: pd.DataFrame,
        aux_match_cols: List[str],
    ) -> tuple[List[str], pd.DataFrame, pd.DataFrame]:
        """Build the covariate matrix and align rows kept for modeling."""
        if self.formula:
            return self._prepare_formula_design_data(data)

        potential_xvars = [col for col in data.columns if col not in self.exclude and col not in aux_match_cols]
        xvars = []
        for col in potential_xvars:
            if data[col].nunique(dropna=False) > 1:
                xvars.append(col)

        filtered_data = data.dropna(subset=xvars).copy() if xvars else data.copy()
        design_matrix = filtered_data[xvars].copy() if xvars else pd.DataFrame(index=filtered_data.index)
        return xvars, filtered_data, design_matrix

    def _prepare_formula_design_data(self, data: pd.DataFrame) -> tuple[List[str], pd.DataFrame, pd.DataFrame]:
        """Build the design matrix from a Patsy-style formula."""
        from patsy import PatsyError, dmatrices, dmatrix

        try:
            if "~" in self.formula:
                lhs = self.formula.split("~", 1)[0].strip()
                if lhs and lhs != self.treatment_col:
                    raise ValueError(
                        f"Formula outcome '{lhs}' must match yvar '{self.treatment_col}'."
                    )
                _, design_matrix = dmatrices(
                    self.formula,
                    data,
                    return_type="dataframe",
                    NA_action="drop",
                )
            else:
                design_matrix = dmatrix(
                    self.formula,
                    data,
                    return_type="dataframe",
                    NA_action="drop",
                )
        except PatsyError as exc:
            raise ValueError(f"Invalid formula '{self.formula}': {exc}") from exc

        design_matrix = design_matrix.drop(columns=["Intercept"], errors="ignore")
        if not design_matrix.empty:
            design_matrix = design_matrix.loc[:, design_matrix.nunique(dropna=False) > 1]

        if design_matrix.empty:
            raise ValueError(
                f"Formula '{self.formula}' did not produce any non-constant covariates."
            )

        filtered_data = data.loc[design_matrix.index].copy()
        return design_matrix.columns.tolist(), filtered_data, design_matrix.copy()


    def fit_model(self, index: int, X: pd.DataFrame, y: pd.Series, model_type: str,
                  balance: bool, max_iter: int = 100,
                  balance_strategy: str = "over",
                  random_state: Optional[int] = None) -> Dict[str, Any]:
        """
        Fits a single propensity score model.

        Internal helper method that calls `pysmatch.modeling.fit_model`.
        Args:
            index (int): An identifier for the model.
            X (pd.DataFrame): The feature matrix (covariates).
            y (pd.Series): The target variable (treatment indicator).
            model_type (str): The type of model to fit ('linear', 'knn', or 'tree').
            balance (bool): Whether the fitting process should aim to balance covariates.
            max_iter (int, optional): Maximum iterations for iterative solvers. Defaults to 100.
            balance_strategy (str, optional): Class-balancing strategy, "over" or "under".
            random_state (int, optional): Base seed for this model; defaults to the matcher's seed.

        Returns:
            Dict[str, Any]: A dictionary containing the fitted model and its accuracy.
        """
        return modeling.fit_model(
            index,
            X,
            y,
            model_type,
            balance,
            max_iter=max_iter,
            random_state=self.random_state if random_state is None else random_state,
            balance_strategy=balance_strategy,
        )

    def fit_scores(self, balance: bool = True, nmodels: Optional[int] = None,
                   n_jobs: int = 1, model_type: str = 'linear',
                   max_iter: int = 100, use_optuna: bool = False,
                   n_trials: int = 10, balance_strategy: str = "over",
                   random_state: Optional[int] = None) -> "Matcher":
        """
        Fits propensity score model(s) to estimate scores.

        Args:
            balance (bool, optional): If True, attempts to create balanced models. Defaults to True.
            nmodels (Optional[int], optional): Number of models for ensemble. Auto-estimated if None and balance=True.
            n_jobs (int, optional): Number of parallel jobs for ensemble fitting. Defaults to 1.
            model_type (str, optional): Type of model ('linear', 'knn', or 'tree' for CatBoost). Defaults to 'linear'.
            max_iter (int, optional): Max iterations for solver. Defaults to 100.
            use_optuna (bool, optional): If True, use Optuna for hyperparameter tuning. Defaults to False.
            n_trials (int, optional): Number of Optuna trials if use_optuna is True. Defaults to 10.
            balance_strategy (str, optional): Balancing strategy when balance=True.
                                              Options: "over" or "under". Defaults to "over".
            random_state (int, optional): Seed used for train/validation splits and model
                                         fitting. Defaults to the value supplied at init.

        Returns:
            Matcher: The current matcher, allowing fluent workflows.
        """
        from multiprocessing import cpu_count
        from multiprocessing.pool import ThreadPool

        self.models, self.model_accuracy = [], []
        self.model_metrics = []
        self.validation_y = None
        self.validation_scores = None
        if random_state is not None:
            self.random_state = random_state
        self.model_type = model_type
        if model_type not in {"linear", "knn", "tree"}:
            raise ValueError("Invalid model_type. Expected one of: 'linear', 'knn', 'tree'.")
        if n_trials < 1:
            raise ValueError("n_trials must be at least 1.")
        if nmodels is not None and nmodels < 1:
            raise ValueError("nmodels must be at least 1.")
        if balance_strategy not in {"over", "under"}:
            raise ValueError(
                f"Invalid balance_strategy='{balance_strategy}'. Expected 'over' or 'under'."
            )
        num_cores = cpu_count()
        n_jobs = min(num_cores, n_jobs if n_jobs > 0 else num_cores)
        logging.info(f"This computer has: {num_cores} cores, using {n_jobs} workers for fitting scores.")

        if self.X.empty or self.y.empty:
            raise ValueError(
                "Feature matrix (X) or target vector (y) is empty; "
                "provide at least one non-constant covariate and both groups."
            )
        if self.y.nunique(dropna=True) < 2:
            raise ValueError("Both treatment groups must be present to fit propensity scores.")

        if use_optuna:
            logging.info(f"Using Optuna for hyperparameter tuning with {n_trials} trials for model_type='{model_type}'.")
            result = modeling.optuna_tuner(self.X, self.y, model_type=model_type,
                                           n_trials=n_trials, balance=balance,
                                           random_state=self.random_state,
                                           balance_strategy=balance_strategy)
            self.models.append(result['model'])
            self.model_accuracy.append(result['accuracy'])
            self.model_metrics.append(result.get('metrics', {"accuracy": result['accuracy']}))
            self.validation_y = np.asarray(result.get('validation_y', []))
            self.validation_scores = np.asarray(result.get('validation_scores', []))
            logging.info(f"[Optuna] Best Accuracy: {result['accuracy']:.2%}")
            self.nmodels = 1
            return self

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
                result = self.fit_model(
                    0,
                    self.X,
                    self.y,
                    model_type,
                    balance=False,
                    max_iter=max_iter,
                    balance_strategy=balance_strategy,
                    random_state=self.random_state,
                ) # balance=False if one class is empty
                self.models.append(result['model'])
                self.model_accuracy.append(result['accuracy'])
                self.model_metrics.append(result.get('metrics', {"accuracy": result['accuracy']}))
                self.validation_y = np.asarray(result.get('validation_y', []))
                self.validation_scores = np.asarray(result.get('validation_scores', []))
                self.nmodels = 1
            else:
                with ThreadPool(n_jobs) as pool:
                    tasks = [
                        (i, self.X, self.y, model_type, balance, max_iter,
                         balance_strategy, self.random_state)
                        for i in range(self.nmodels)
                    ]
                    results = pool.starmap(self.fit_model, tasks)
                for res in results:
                    self.models.append(res['model'])
                    self.model_accuracy.append(res['accuracy'])
                    self.model_metrics.append(res.get('metrics', {"accuracy": res['accuracy']}))
                if results:
                    self.validation_y = np.asarray(results[0].get('validation_y', []))
                    self.validation_scores = np.asarray(results[0].get('validation_scores', []))
        else:
            result = self.fit_model(
                0,
                self.X,
                self.y,
                model_type,
                balance,
                max_iter,
                balance_strategy,
                self.random_state,
            )
            self.models.append(result['model'])
            self.model_accuracy.append(result['accuracy'])
            self.model_metrics.append(result.get('metrics', {"accuracy": result['accuracy']}))
            self.validation_y = np.asarray(result.get('validation_y', []))
            self.validation_scores = np.asarray(result.get('validation_scores', []))

        if self.model_accuracy:
            if self.nmodels > 1:
                avg_accuracy = np.mean(self.model_accuracy)
                logging.info(f"Average Accuracy over {self.nmodels} models: {avg_accuracy:.2%}")
            else:
                logging.info(f"Accuracy for single model: {self.model_accuracy[0]*100:.2f}%")
        else:
            logging.warning("No model accuracies recorded.")

        return self


    def predict_scores(self, batch_size: Optional[int] = None) -> "Matcher":
        """
        Predicts propensity scores using the fitted model(s).
        Scores are stored in `self.data['scores']`.

        Args:
            batch_size (int, optional): Number of rows scored per model call.
                                       ``None`` preserves the original all-at-once
                                       behavior.

        Returns:
            Matcher: The current matcher, allowing fluent workflows.
        """
        if not self.models:
            raise RuntimeError("No trained models found; call fit_scores() before predict_scores().")
        if self.X.empty:
            raise ValueError("Feature matrix (X) is empty; cannot predict propensity scores.")
        if batch_size is not None and (not isinstance(batch_size, (int, np.integer)) or batch_size < 1):
            raise ValueError("batch_size must be a positive integer or None.")

        try:
            scores_chunks = []
            step = batch_size or len(self.X)
            for start in range(0, len(self.X), step):
                X_pred = self.X.iloc[start:start + step].copy()
                for col in X_pred.columns:
                    if X_pred[col].dtype == 'object' or pd.api.types.is_string_dtype(X_pred[col]):
                        try:
                            X_pred[col] = pd.to_numeric(X_pred[col], errors='raise')
                        except ValueError:
                            # Keep genuine categorical values for the fitted
                            # preprocessing pipeline (OneHotEncoder/CatBoost).
                            logging.debug(
                                "Keeping non-numeric categorical column '%s' for model '%s'.",
                                col,
                                self.model_type,
                            )
                model_preds = [model.predict_proba(X_pred)[:, 1] for model in self.models]
                scores_chunks.append(np.mean(model_preds, axis=0))
            scores = np.concatenate(scores_chunks) if scores_chunks else np.array([], dtype=float)
            self.data['scores'] = scores
            # Update self.test_df and self.control_df (and self.test, self.control) to include scores
            self.test_df['scores'] = self.data.loc[self.test_df.index, 'scores']
            self.control_df['scores'] = self.data.loc[self.control_df.index, 'scores']
            self.test = self.test_df.copy()
            self.control = self.control_df.copy()
            logging.info("Propensity scores predicted and added to 'scores' column in self.data, self.test_df, self.control_df.")
            return self
        except Exception as e:
            raise RuntimeError(
                "Could not predict propensity scores. Check that the covariates have the "
                f"same schema used during fitting (model_type='{self.model_type}')."
            ) from e


    def match(self, threshold: float = 0.001, nmatches: int = 1, method: str = 'min',
              replacement: bool = False, exhaustive_matching: Optional[bool] = None,
              random_state: Optional[int] = None,
              distance: str = "probability",
              exact: Optional[List[str]] = None,
              chunk_size: Optional[int] = None) -> pd.DataFrame:
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
            random_state (int, optional): Seed used when ``method="random"``. Defaults to
                                         the matcher's configured seed.
            distance (str, optional): ``"probability"`` or ``"logit"`` propensity
                                     distance. Defaults to ``"probability"``.
            exact (list[str], optional): Covariates that must match exactly.
            chunk_size (int, optional): Number of treated rows processed per
                                       matching query. Defaults to ``None``.

        Returns:
            pandas.DataFrame: The matched rows. An empty frame means that no pair met
                              the requested threshold.
        """
        if exhaustive_matching is None:
            exhaustive_matching = self.exhaustive_matching_default

        if 'scores' not in self.data.columns:
            self.matched_data = pd.DataFrame()
            raise RuntimeError("Propensity scores are missing; call predict_scores() before match().")

        if 'record_id' not in self.data.columns:
            logging.error("'record_id' column not found in self.data. This is unexpected.")
            self.data['record_id'] = self.data.index # Fallback

        # Use self.test and self.control which should have scores if predict_scores was called
        current_test_df = self.test.copy() # self.test is self.test_df updated with scores
        current_control_df = self.control.copy() # self.control is self.control_df updated with scores


        if current_test_df.empty or current_control_df.empty:
            self.matched_data = pd.DataFrame()
            raise ValueError("Both treated and control groups must contain at least one row.")

        if 'scores' not in current_test_df.columns or 'scores' not in current_control_df.columns:
            self.matched_data = pd.DataFrame()
            raise RuntimeError("Propensity scores are missing from one of the groups; rerun predict_scores().")


        if exhaustive_matching:
            logging.info(f"Performing exhaustive matching: nmatches={nmatches}, threshold={threshold}")
            if 'record_id' not in current_test_df.columns or 'record_id' not in current_control_df.columns:
                self.matched_data = pd.DataFrame()
                raise RuntimeError("Record IDs are missing from the processed groups.")

            self.matched_data = matching.perform_exhaustive_match(
                self.data,
                self.treatment_col,
                threshold=threshold,
                nmatches=nmatches,
                show_progress=True,
                distance=distance,
                exact=exact,
                chunk_size=chunk_size,
            )

            if self.matched_data.empty:
                logging.info("No matches found with exhaustive matching.")
                return self.matched_data
            logging.info(f"Exhaustive matching complete. {self.matched_data['match_id'].nunique() if not self.matched_data.empty else 0} pairs formed.")

        else:
            logging.info(f"Performing matching using pysmatch.matching.perform_match: method='{method}', replacement={replacement}, threshold={threshold}, nmatches={nmatches}")
            if 'scores' not in self.data.columns or 'record_id' not in self.data.columns or self.treatment_col not in self.data.columns:
                self.matched_data = pd.DataFrame()
                raise RuntimeError("Processed data is missing a required matching column.")

            self.matched_data = matching.perform_match(
                self.data,
                self.treatment_col,
                threshold=threshold,
                nmatches=nmatches,
                method=method,
                replacement=replacement,
                random_state=self.random_state if random_state is None else random_state,
                distance=distance,
                exact=exact,
                chunk_size=chunk_size,
            )
            if self.matched_data.empty:
                logging.info("No matches found using pysmatch.matching.perform_match.")
            else:
                logging.info(f"Matching with pysmatch.matching.perform_match complete. Matched data has {len(self.matched_data)} rows.")
                if 'record_id' not in self.matched_data.columns:
                    logging.warning("'record_id' not found in matched_data from matching.perform_match. Subsequent weighting might fail.")
                if 'match_id' not in self.matched_data.columns:
                    logging.warning("'match_id' not found in matched_data from matching.perform_match. Record frequency might be affected.")

        return self.matched_data


    def plot_scores(self) -> None:
        """
        Plots the distribution of propensity scores before matching.
        Visualizes score overlap between test (treated) and control groups.
        """
        if 'scores' not in self.data.columns:
            logging.warning("Scores not available in self.data. Cannot plot scores. Run predict_scores() first.")
            return
        visualization = _visualization_module()
        visualization.plot_scores(self.data, self.treatment_col,
                                  control_color=self.control_color,
                                  test_color=self.test_color)

    def plot_love(self, threshold: float = 0.1, ax=None):
        """Plot absolute SMDs before and after matching."""

        return _visualization_module().plot_love(self, threshold=threshold, ax=ax)

    def plot_overlap(self, scope: str = "before", ax=None):
        """Plot treated/control propensity-score overlap."""

        if scope not in {"before", "after"}:
            raise ValueError("scope must be 'before' or 'after'.")
        return _visualization_module().plot_overlap(self, scope=scope, ax=ax)

    def plot_calibration(self, ax=None):
        """Plot validation propensity-score calibration."""

        return _visualization_module().plot_calibration(self, ax=ax)

    def tune_threshold(self, method: str, nmatches: int = 1,
                       rng: Optional[np.ndarray] = None,
                       random_state: Optional[int] = None,
                       distance: str = "probability",
                       exact: Optional[List[str]] = None,
                       chunk_size: Optional[int] = None) -> tuple[np.ndarray, list[float]]:
        """
        Evaluates matching retention across a range of threshold values.

        Args:
            method (str): Matching method ('min' or 'random') for evaluation.
            nmatches (int, optional): Number of matches per treated sample. Defaults to 1.
            rng (Optional[np.ndarray], optional): Threshold values to test. Defaults to np.arange(0, 0.001, 0.0001).
            random_state (int, optional): Seed used for random matching during evaluation.
                                         Defaults to the matcher's configured seed.
            distance (str, optional): ``"probability"`` or ``"logit"`` score distance.
            exact (list[str], optional): Exact-matching columns.
            chunk_size (int, optional): Treated-row chunk size for each threshold.

        Returns:
            tuple: The tested thresholds and retained proportions.
        """
        if 'scores' not in self.data.columns:
            raise RuntimeError("Propensity scores are missing; call predict_scores() before tune_threshold().")
        if rng is None:
            rng = np.arange(0, 0.0011, 0.0001)

        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as exc:
            raise ImportError(
                "Threshold plots require the optional visualization dependencies; "
                "install them with `pip install 'pysmatch[viz]'`."
            ) from exc

        thresholds, retained = matching.tune_threshold(self.data, self.treatment_col,
                                                       method=method, nmatches=nmatches, rng=rng,
                                                       random_state=self.random_state if random_state is None else random_state,
                                                       distance=distance, exact=exact,
                                                       chunk_size=chunk_size)
        plt.figure(figsize=(10,6))
        plt.plot(thresholds, retained, marker='o')
        plt.title("Proportion of Minority Group Retained for Threshold Grid")
        plt.ylabel("Proportion Retained (Minority Group)")
        plt.xlabel("Threshold")
        plt.xticks(thresholds, rotation=45, ha="right")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
        return thresholds, retained

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
        elif 'match_id' in self.matched_data.columns and self.treatment_col in self.matched_data.columns:
            control_records = self.matched_data[self.matched_data[self.treatment_col] == 0]
            if control_records.empty:
                return pd.DataFrame(columns=['record_id', 'n_occurrences_as_control'])
            freqs = control_records['record_id'].value_counts().rename_axis('record_id').reset_index(name='n_occurrences_as_control')
            return freqs
        else:
            logging.warning("Could not determine record frequency due to missing 'matched_as' or 'match_id' in matched_data.")
            return pd.DataFrame(columns=['record_id', 'n_occurrences_as_control'])


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

    def balance_report(self) -> pd.DataFrame:
        """Return a compact covariate-balance report before and after matching.

        Numeric covariates use the standardized mean difference (SMD). For
        categorical covariates, ``smd_*`` is the largest absolute difference
        in category proportions. Values near zero indicate better balance;
        the report deliberately avoids treating p-values as a balance target.
        """

        before = self.data
        after = self.matched_data
        rows: list[dict[str, Any]] = []

        def numeric_smd(treated: pd.Series, control: pd.Series) -> float:
            t = pd.to_numeric(treated, errors="coerce").dropna().to_numpy()
            c = pd.to_numeric(control, errors="coerce").dropna().to_numpy()
            if len(t) == 0 or len(c) == 0:
                return float("nan")
            pooled_var = (np.var(t, ddof=1) + np.var(c, ddof=1)) / 2
            if pooled_var == 0:
                return 0.0 if np.mean(t) == np.mean(c) else float("inf")
            return float((np.mean(t) - np.mean(c)) / np.sqrt(pooled_var))

        def categorical_smd(treated: pd.Series, control: pd.Series) -> float:
            t_props = treated.dropna().value_counts(normalize=True)
            c_props = control.dropna().value_counts(normalize=True)
            if t_props.empty or c_props.empty:
                return float("nan")
            levels = t_props.index.union(c_props.index)
            return float((t_props.reindex(levels, fill_value=0)
                          - c_props.reindex(levels, fill_value=0)).abs().max())

        for col in self.xvars:
            if col not in before.columns:
                # Formula design matrices may contain generated columns rather
                # than a same-named source column.
                continue
            before_t = before.loc[before[self.treatment_col] == 1, col]
            before_c = before.loc[before[self.treatment_col] == 0, col]
            if after.empty or self.treatment_col not in after.columns or col not in after.columns:
                after_t = pd.Series(dtype=before[col].dtype)
                after_c = pd.Series(dtype=before[col].dtype)
            else:
                after_t = after.loc[after[self.treatment_col] == 1, col]
                after_c = after.loc[after[self.treatment_col] == 0, col]

            is_numeric = pd.api.types.is_numeric_dtype(before[col])
            smd_before = numeric_smd(before_t, before_c) if is_numeric else categorical_smd(before_t, before_c)
            smd_after = numeric_smd(after_t, after_c) if is_numeric else categorical_smd(after_t, after_c)
            rows.append({
                "variable": col,
                "type": "numeric" if is_numeric else "categorical",
                "smd_before": smd_before,
                "smd_after": smd_after,
                "abs_smd_before": abs(smd_before) if not np.isnan(smd_before) else np.nan,
                "abs_smd_after": abs(smd_after) if not np.isnan(smd_after) else np.nan,
                "treated_before": int(before_t.notna().sum()),
                "control_before": int(before_c.notna().sum()),
                "treated_after": int(after_t.notna().sum()),
                "control_after": int(after_c.notna().sum()),
            })

        return pd.DataFrame(rows, columns=[
            "variable", "type", "smd_before", "smd_after",
            "abs_smd_before", "abs_smd_after", "treated_before",
            "control_before", "treated_after", "control_after",
        ])

    def balance_summary(self) -> Dict[str, Any]:
        """Return maximum and mean absolute SMD values for quick review."""

        report = self.balance_report()

        def finite_max(series: pd.Series) -> float:
            values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            return float(values.max()) if not values.empty else float("nan")

        def finite_mean(series: pd.Series) -> float:
            values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            return float(values.mean()) if not values.empty else float("nan")

        return {
            "n_covariates": int(len(report)),
            "max_abs_smd_before": finite_max(report.get("abs_smd_before", pd.Series(dtype=float))),
            "max_abs_smd_after": finite_max(report.get("abs_smd_after", pd.Series(dtype=float))),
            "mean_abs_smd_before": finite_mean(report.get("abs_smd_before", pd.Series(dtype=float))),
            "mean_abs_smd_after": finite_mean(report.get("abs_smd_after", pd.Series(dtype=float))),
        }

    def score_report(self) -> pd.DataFrame:
        """Return validation discrimination and calibration metrics per model."""

        if not self.model_metrics:
            raise RuntimeError("No validation metrics found; call fit_scores() first.")
        rows = []
        for index, metrics in enumerate(self.model_metrics, start=1):
            rows.append({"model": index, **metrics})
        return pd.DataFrame(rows, columns=["model", "accuracy", "auc", "brier"])

    def calibration_report(self, n_bins: int = 10) -> pd.DataFrame:
        """Return binned validation calibration statistics."""

        if self.validation_y is None or self.validation_scores is None or len(self.validation_y) == 0:
            raise RuntimeError("No validation predictions found; call fit_scores() first.")
        if not isinstance(n_bins, (int, np.integer)) or not 2 <= n_bins <= 50:
            raise ValueError("n_bins must be an integer between 2 and 50.")
        y = np.asarray(self.validation_y, dtype=float)
        scores = np.clip(np.asarray(self.validation_scores, dtype=float), 0, 1)
        bins = np.linspace(0, 1, n_bins + 1)
        assignments = np.minimum(np.digitize(scores, bins[1:-1], right=False), n_bins - 1)
        rows = []
        for index in range(n_bins):
            mask = assignments == index
            if not mask.any():
                continue
            observed = float(np.mean(y[mask]))
            predicted = float(np.mean(scores[mask]))
            rows.append({
                "bin": index,
                "lower": float(bins[index]),
                "upper": float(bins[index + 1]),
                "n": int(mask.sum()),
                "mean_predicted": predicted,
                "observed_rate": observed,
                "calibration_gap": observed - predicted,
            })
        return pd.DataFrame(rows, columns=[
            "bin", "lower", "upper", "n", "mean_predicted", "observed_rate", "calibration_gap"
        ])

    def effective_sample_size(self) -> pd.DataFrame:
        """Report effective sample size (ESS) before and after matching."""

        frames = [("before", self.data)]
        if not self.matched_data.empty:
            frames.append(("after", self.matched_data))
        rows = []
        for scope, frame in frames:
            if self.treatment_col not in frame.columns:
                continue
            for group, label in [(1, "treated"), (0, "control")]:
                subset = frame[frame[self.treatment_col] == group]
                if subset.empty:
                    rows.append({"scope": scope, "group": label, "n": 0, "weight_sum": 0.0, "ess": 0.0})
                    continue
                if "weight" in subset.columns:
                    weights = pd.to_numeric(subset["weight"], errors="coerce").fillna(0).to_numpy(dtype=float)
                elif "record_id" in subset.columns:
                    counts = subset["record_id"].value_counts()
                    weights = subset["record_id"].map(lambda value: 1.0 / counts[value]).to_numpy(dtype=float)
                else:
                    weights = np.ones(len(subset), dtype=float)
                weight_sum = float(weights.sum())
                ess = float(weight_sum ** 2 / np.square(weights).sum()) if np.square(weights).sum() else 0.0
                rows.append({
                    "scope": scope,
                    "group": label,
                    "n": int(len(subset)),
                    "weight_sum": weight_sum,
                    "ess": ess,
                })
        return pd.DataFrame(rows, columns=["scope", "group", "n", "weight_sum", "ess"])

    def positivity_report(
        self,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
    ) -> pd.DataFrame:
        """Report common-support retention and optional propensity trimming."""

        if "scores" not in self.data.columns:
            raise RuntimeError("Propensity scores are missing; call predict_scores() first.")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("lower must be less than or equal to upper.")
        frames = [("before", self.data)]
        if not self.matched_data.empty:
            frames.append(("after", self.matched_data))
        rows = []
        for scope, frame in frames:
            if "scores" not in frame.columns or self.treatment_col not in frame.columns:
                continue
            treated = pd.to_numeric(frame.loc[frame[self.treatment_col] == 1, "scores"], errors="coerce").dropna()
            control = pd.to_numeric(frame.loc[frame[self.treatment_col] == 0, "scores"], errors="coerce").dropna()
            if treated.empty or control.empty:
                continue
            common_lower = max(float(treated.min()), float(control.min()))
            common_upper = min(float(treated.max()), float(control.max()))
            trim_lower = common_lower if lower is None else float(lower)
            trim_upper = common_upper if upper is None else float(upper)
            treated_keep = ((treated >= trim_lower) & (treated <= trim_upper)).sum()
            control_keep = ((control >= trim_lower) & (control <= trim_upper)).sum()
            rows.append({
                "scope": scope,
                "common_lower": common_lower,
                "common_upper": common_upper,
                "trim_lower": trim_lower,
                "trim_upper": trim_upper,
                "treated_n": int(len(treated)),
                "control_n": int(len(control)),
                "treated_kept": int(treated_keep),
                "control_kept": int(control_keep),
                "treated_trimmed": int(len(treated) - treated_keep),
                "control_trimmed": int(len(control) - control_keep),
                "treated_retention": float(treated_keep / len(treated)),
                "control_retention": float(control_keep / len(control)),
            })
        return pd.DataFrame(rows)

    def estimate_effect(
        self,
        outcome: str,
        estimand: str = "ATT",
        bootstrap: int = 1000,
        alpha: float = 0.05,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Estimate a mean outcome difference from the matched sample.

        The method uses unique matched units as the analysis units. For ATT,
        each treated unit is compared with the mean outcome of its matched
        controls. ATC reverses that direction. ATE combines the unit-level
        ATT and ATC contrasts over the matched sample. This is a descriptive
        matched-sample estimate; it does not remove unmeasured confounding.

        Args:
            outcome: Numeric outcome column in ``matched_data``.
            estimand: ``"ATT"``, ``"ATC"``, or ``"ATE"``.
            bootstrap: Number of unit-level bootstrap draws for the confidence
                       interval. Set to ``0`` to skip resampling.
            alpha: Two-sided confidence level cutoff, between 0 and 1.
            random_state: Seed for bootstrap resampling. Defaults to the
                          matcher's configured seed.

        Returns:
            A dictionary containing the estimate, standard error, percentile
            confidence interval, and matched-unit counts.

        Raises:
            RuntimeError: If matching has not been performed.
            ValueError: If the outcome, estimand, or bootstrap settings are
                        invalid, or if a matched outcome is missing/non-numeric.
        """

        if self.matched_data.empty:
            raise RuntimeError("No matched data found; call match() before estimate_effect().")
        if outcome not in self.matched_data.columns:
            raise ValueError(f"Outcome column '{outcome}' was not found in matched_data.")
        estimand = estimand.upper()
        if estimand not in {"ATT", "ATC", "ATE"}:
            raise ValueError("estimand must be one of 'ATT', 'ATC', or 'ATE'.")
        if not isinstance(bootstrap, (int, np.integer)) or bootstrap < 0:
            raise ValueError("bootstrap must be a non-negative integer.")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1.")
        if self.treatment_col not in self.matched_data.columns:
            raise ValueError(f"Treatment column '{self.treatment_col}' is missing from matched_data.")
        if "match_id" not in self.matched_data.columns:
            raise ValueError("matched_data must contain 'match_id' to estimate an effect.")

        data = self.matched_data.copy()
        data[outcome] = pd.to_numeric(data[outcome], errors="coerce")
        if data[outcome].isna().any():
            raise ValueError(f"Outcome column '{outcome}' must contain only numeric, non-missing values.")

        record_id = "record_id" if "record_id" in data.columns else None
        if record_id is None:
            data["record_id"] = data.index
            record_id = "record_id"

        # A standard nmatches result may repeat a treated row inside one
        # match_id. Deduplicate each unit within a match before constructing
        # unit-level contrasts.
        data = data.drop_duplicates(subset=["match_id", record_id], keep="first")
        treated = data[data[self.treatment_col] == 1]
        control = data[data[self.treatment_col] == 0]
        if treated.empty or control.empty:
            raise ValueError("Both treated and control units are required in matched_data.")

        def unit_effects(target: pd.DataFrame, counterfactual: pd.DataFrame) -> np.ndarray:
            target_values = target.groupby(record_id, sort=False)[outcome].first()
            links = target[[record_id, "match_id"]].drop_duplicates()
            counterfactual_by_match = counterfactual.groupby("match_id")[outcome].mean()
            links["counterfactual"] = links["match_id"].map(counterfactual_by_match)
            links = links.dropna(subset=["counterfactual"])
            if links.empty:
                return np.array([], dtype=float)
            counterfactual_values = links.groupby(record_id, sort=False)["counterfactual"].mean()
            common_ids = target_values.index.intersection(counterfactual_values.index)
            return (target_values.loc[common_ids] - counterfactual_values.loc[common_ids]).to_numpy(dtype=float)

        att_effects = unit_effects(treated, control)
        # ATC targets controls but keeps the treatment-effect convention
        # (treated outcome minus control outcome).
        atc_effects = -unit_effects(control, treated)
        if estimand == "ATT":
            effects = att_effects
        elif estimand == "ATC":
            effects = atc_effects
        else:
            effects = np.concatenate([att_effects, atc_effects])
        if len(effects) == 0:
            raise ValueError("No complete treated/control matched units were available for the requested estimand.")

        estimate = float(np.mean(effects))
        std_error = float(np.std(effects, ddof=1) / np.sqrt(len(effects))) if len(effects) > 1 else float("nan")
        ci_lower = ci_upper = float("nan")
        if bootstrap:
            seed = self.random_state if random_state is None else random_state
            rng = np.random.default_rng(seed)
            draws = np.empty(bootstrap, dtype=float)
            for index in range(bootstrap):
                draws[index] = np.mean(rng.choice(effects, size=len(effects), replace=True))
            ci_lower, ci_upper = np.quantile(draws, [alpha / 2, 1 - alpha / 2]).astype(float)

        matched_pairs = int(data["match_id"].nunique())
        return {
            "outcome": outcome,
            "estimand": estimand,
            "estimate": estimate,
            "std_error": std_error,
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "bootstrap": int(bootstrap),
            "alpha": float(alpha),
            "n_units": int(len(effects)),
            "n_treated": int(att_effects.size),
            "n_control": int(atc_effects.size),
            "n_pairs": matched_pairs,
        }

    def overlap_report(self) -> pd.DataFrame:
        """Summarize propensity-score overlap before and after matching.

        The common-support interval is the intersection of the treated and
        control score ranges. Counts inside that interval make it easier to
        detect positivity problems before interpreting an effect estimate.
        """

        if "scores" not in self.data.columns:
            raise RuntimeError("Propensity scores are missing; call predict_scores() first.")

        frames = [("before", self.data)]
        if not self.matched_data.empty:
            frames.append(("after", self.matched_data))
        rows: list[dict[str, Any]] = []
        for scope, frame in frames:
            if "scores" not in frame.columns or self.treatment_col not in frame.columns:
                continue
            treated = pd.to_numeric(
                frame.loc[frame[self.treatment_col] == 1, "scores"], errors="coerce"
            ).dropna()
            control = pd.to_numeric(
                frame.loc[frame[self.treatment_col] == 0, "scores"], errors="coerce"
            ).dropna()
            if treated.empty or control.empty:
                rows.append({"scope": scope, "treated_n": int(len(treated)), "control_n": int(len(control))})
                continue
            lower = max(float(treated.min()), float(control.min()))
            upper = min(float(treated.max()), float(control.max()))
            rows.append({
                "scope": scope,
                "treated_n": int(len(treated)),
                "control_n": int(len(control)),
                "treated_min": float(treated.min()),
                "treated_max": float(treated.max()),
                "control_min": float(control.min()),
                "control_max": float(control.max()),
                "common_lower": lower,
                "common_upper": upper,
                "common_width": max(0.0, upper - lower),
                "treated_in_common": int(((treated >= lower) & (treated <= upper)).sum()),
                "control_in_common": int(((control >= lower) & (control <= upper)).sum()),
            })
        return pd.DataFrame(rows)


    def compare_continuous(self, save: bool = False, return_table: bool = False, plot_result: bool = True):
        """
        Compares continuous variables between groups before and after matching.
        Delegates to `visualization.compare_continuous`.
        The `visualization` module is expected to use `self.yvar` (which is `self.treatment_col`).
        """
        visualization = _visualization_module()
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
        visualization = _visualization_module()
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

        visualization = _visualization_module()
        visualization.plot_matched_scores(
            self.matched_data,
            self.treatment_col,
            control_color=self.control_color,
            test_color=self.test_color
        )
