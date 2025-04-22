# modeling.py
# -*- coding: utf-8 -*-
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any

def fit_model(index: int, X: pd.DataFrame, y: pd.Series, model_type: str,
              balance: bool, max_iter: int = 100, random_state: int = 42) -> Dict[str, Any]:
    """
    Fits a single propensity score model with preprocessing.

    This function trains a specified classification model (Logistic Regression,
    CatBoost, or KNN) to predict the binary target variable `y` based on covariates `X`.
    It includes steps for:
    1. Splitting data into training and a (currently unused) test set.
    2. Optionally balancing the training data using RandomOverSampler.
    3. Preprocessing features using StandardScaler for numerical and OneHotEncoder
       for categorical variables via a ColumnTransformer.
    4. Training the specified model within a scikit-learn Pipeline that combines
       preprocessing and classification.
    5. Calculating the accuracy of the fitted model on the (potentially resampled)
       training data.

    Args:
        index (int): An identifier for this model instance (e.g., used for setting
                     random states in cross-validation or ensembles).
        X (pd.DataFrame): DataFrame containing the covariate features.
        y (pd.Series): Series containing the binary target variable (0 or 1).
        model_type (str): The type of classifier to use. Options: 'linear'
                          (Logistic Regression), 'tree' (CatBoostClassifier),
                          'knn' (KNeighborsClassifier).
        balance (bool): If True, applies RandomOverSampler to the training data
                        to balance the classes before fitting the model.
        max_iter (int, optional): Maximum number of iterations for solvers in
                                  iterative models (like Logistic Regression).
                                  Defaults to 100.
        random_state (Optional[int], optional): Seed for reproducibility in data splitting,
                                     resampling (if `balance=True`), and model initialization
                                     (where applicable, like LogisticRegression, CatBoost).
                                     Defaults to None. If None, randomness is not fixed.
                                     *Note: Uses `index` for train_test_split and sampler seed.*


    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'model': The fitted scikit-learn Pipeline object (preprocessor + classifier).
            - 'accuracy': The accuracy score of the fitted model on the (potentially
                          resampled) training data.

    Raises:
        ValueError: If an invalid `model_type` is provided (not 'linear', 'tree', or 'knn').
        ImportError: If required libraries (sklearn, imblearn, catboost) are not installed.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.pipeline import Pipeline

    X_train, _, y_train, _ = train_test_split(X, y, train_size=0.7, random_state=index)

    if balance:
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=index)
        X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
    else:
        X_resampled, y_resampled = X_train, y_train

    numerical_features = X_resampled.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_resampled.select_dtypes(exclude=np.number).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    if model_type == 'tree':
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(
            iterations=max_iter,
            depth=6,
            eval_metric='AUC',
            l2_leaf_reg=3,
            learning_rate=0.02,
            loss_function='Logloss',
            logging_level='Silent',
            random_seed=random_state
        )
    elif model_type == 'knn':
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=5)
    elif model_type == 'linear':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=max_iter, random_state=random_state)
    else:
        raise ValueError("Invalid model_type provided.")

    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    y_resampled_series = y_resampled.iloc[:, 0] if isinstance(y_resampled, pd.DataFrame) else y_resampled
    pipeline.fit(X_resampled, y_resampled_series)
    accuracy = pipeline.score(X_resampled, y_resampled_series)

    logging.info(f"Model {index + 1} trained. Accuracy: {accuracy:.2%}")
    return {'model': pipeline, 'accuracy': accuracy}


def optuna_tuner(X: pd.DataFrame, y: pd.Series, model_type: str, n_trials: int = 10,
                 balance: bool = True, random_state: int = 42) -> Dict[str, Any]:
    """
    Performs hyperparameter optimization using Optuna for a specified model type.

    This function uses Optuna to search for the best hyperparameters for a given
    `model_type` ('linear', 'tree', 'knn'). It involves:
    1. Splitting data into training and validation sets.
    2. Optionally balancing the training set using RandomOverSampler.
    3. Defining an Optuna `objective` function that:
        - Takes an Optuna `trial` object.
        - Defines hyperparameter search spaces using `trial.suggest_*`.
        - Creates a preprocessing pipeline (StandardScaler + OneHotEncoder).
        - Initializes the specified model with suggested hyperparameters.
        - Combines preprocessing and model into a scikit-learn Pipeline.
        - Fits the pipeline on the (potentially resampled) training data.
        - Evaluates the pipeline on the validation set and returns the accuracy score.
    4. Running the Optuna study for `n_trials` to maximize the objective (accuracy).
    5. Retrieving the best hyperparameters found by the study.
    6. Training a final pipeline on the (potentially resampled) training data using the
       best hyperparameters found.
    7. Returning the final fitted pipeline, its validation accuracy during the study,
       and the best parameters.

    Args:
        X (pd.DataFrame): DataFrame containing the covariate features.
        y (pd.Series): Series containing the binary target variable (0 or 1).
        model_type (str): The type of classifier to tune. Options: 'linear'
                          (Logistic Regression), 'tree' (CatBoostClassifier),
                          'knn' (KNeighborsClassifier).
        n_trials (int, optional): The number of optimization trials Optuna should run.
                                  Defaults to 10.
        balance (bool, optional): If True, applies RandomOverSampler to the training
                                  data within each Optuna trial before fitting the model.
                                  Defaults to True.
        random_state (int, optional): Seed for reproducibility in data splitting,
                                      resampling, and model initialization (where applicable).
                                      Defaults to 42.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'model': The final scikit-learn Pipeline object fitted with the best
                       hyperparameters found by Optuna on the training data.
            - 'accuracy': The best accuracy score achieved on the validation set during
                          the Optuna study.
            - 'best_params': A dictionary containing the best hyperparameters found.

    Raises:
        ValueError: If an invalid `model_type` is provided for tuning.
        ImportError: If required libraries (optuna, sklearn, imblearn, catboost) are not installed.
    """
    import optuna
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.pipeline import Pipeline

    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.7, random_state=random_state)

    if balance:
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=random_state)
        X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
    else:
        X_resampled, y_resampled = X_train, y_train

    numerical_features = X_resampled.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_resampled.select_dtypes(exclude=np.number).columns.tolist()

    def objective(trial: optuna.trial.Trial) -> float:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )
        if model_type == 'linear':
            from sklearn.linear_model import LogisticRegression
            c_val = trial.suggest_float("C", 1e-3, 1e3, log=True)
            max_iter_trial = trial.suggest_int("max_iter", 100, 500, step=50)
            model = LogisticRegression(C=c_val, max_iter=max_iter_trial, random_state=random_state)
        elif model_type == 'tree':
            from catboost import CatBoostClassifier
            depth = trial.suggest_int("depth", 4, 10)
            learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
            iterations = trial.suggest_int("iterations", 100, 500, step=50)
            model = CatBoostClassifier(
                depth=depth,
                iterations=iterations,
                learning_rate=learning_rate,
                eval_metric='AUC',
                random_seed=random_state,
                logging_level='Silent'
            )
        elif model_type == 'knn':
            from sklearn.neighbors import KNeighborsClassifier
            n_neighbors = trial.suggest_int("n_neighbors", 1, 20)
            model = KNeighborsClassifier(n_neighbors=n_neighbors)
        else:
            raise ValueError("Invalid model_type for optuna tuning.")

        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        pipeline.fit(X_resampled, y_resampled)
        score = pipeline.score(X_val, y_val)
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    best_score = study.best_value

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    if model_type == 'linear':
        from sklearn.linear_model import LogisticRegression
        final_model = LogisticRegression(
            C=best_params.get("C", 1.0),
            max_iter=best_params.get("max_iter", 100),
            random_state=random_state
        )
    elif model_type == 'tree':
        from catboost import CatBoostClassifier
        final_model = CatBoostClassifier(
            depth=best_params.get("depth", 6),
            iterations=best_params.get("iterations", 100),
            learning_rate=best_params.get("learning_rate", 0.02),
            eval_metric='AUC',
            random_seed=random_state,
            logging_level='Silent'
        )
    elif model_type == 'knn':
        from sklearn.neighbors import KNeighborsClassifier
        final_model = KNeighborsClassifier(
            n_neighbors=best_params.get("n_neighbors", 5)
        )
    else:
        raise ValueError("Invalid model_type for final pipeline.")

    final_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', final_model)])
    final_pipeline.fit(X_resampled, y_resampled)
    logging.info(f"[Optuna] Best params: {best_params}, best score: {best_score:.2%}")

    return {'model': final_pipeline, 'accuracy': best_score, 'best_params': best_params}