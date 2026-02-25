# modeling.py
# -*- coding: utf-8 -*-
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Literal, Tuple

BalanceStrategy = Literal["over", "under"]


def _rebalance_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    balance: bool,
    balance_strategy: BalanceStrategy,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Resample the training set if balancing is enabled."""
    if not balance:
        return X_train, y_train

    if balance_strategy == "over":
        from imblearn.over_sampling import RandomOverSampler

        sampler = RandomOverSampler(random_state=random_state)
    elif balance_strategy == "under":
        from imblearn.under_sampling import RandomUnderSampler

        sampler = RandomUnderSampler(random_state=random_state)
    else:
        raise ValueError(
            f"Invalid balance_strategy='{balance_strategy}'. Expected 'over' or 'under'."
        )

    X_resampled_np, y_resampled_np = sampler.fit_resample(X_train, y_train)
    X_resampled = pd.DataFrame(X_resampled_np, columns=X_train.columns)
    y_resampled = pd.Series(
        y_resampled_np,
        name=y_train.name if hasattr(y_train, "name") else "target",
    )
    return X_resampled, y_resampled

def fit_model(index: int, X: pd.DataFrame, y: pd.Series, model_type: str,
              balance: bool, max_iter: int = 100, random_state: int = 42,
              balance_strategy: BalanceStrategy = "over") -> Dict[str, Any]:
    """
    Fit a single propensity-score model and evaluate it on a held-out split.

    Returns a fitted pipeline and validation-set accuracy.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.pipeline import Pipeline

    if not isinstance(y, pd.Series):
        y = pd.Series(y, name="target")

    seed = random_state + index
    stratify = y if y.nunique() > 1 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        train_size=0.7,
        random_state=seed,
        stratify=stratify,
    )

    X_resampled, y_resampled = _rebalance_training_data(
        X_train=X_train,
        y_train=y_train,
        balance=balance,
        balance_strategy=balance_strategy,
        random_state=seed,
    )

    original_numerical_features = X_resampled.select_dtypes(include='number').columns.tolist()
    original_categorical_features = X_resampled.select_dtypes(exclude='number').columns.tolist()

    pipeline: Pipeline # Define type for pipeline

    if model_type == 'tree':
        from catboost import CatBoostClassifier

        # Preprocessor for CatBoost: scale numericals, pass through categoricals.
        catboost_preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), original_numerical_features)
            ],
            remainder='passthrough' # Categorical features will be passed through
        )

        cat_feature_indices_for_cb = []
        if original_categorical_features: # Only if there are categorical features
            # Calculate indices of categorical features AFTER ColumnTransformer
            # Numerical features come first, then passthrough categorical features
            cat_feature_indices_for_cb = [
                len(original_numerical_features) + i
                for i in range(len(original_categorical_features))
            ]
            logging.debug(f"Model {index+1} (CatBoost): Numerical features: {original_numerical_features}")
            logging.debug(f"Model {index+1} (CatBoost): Categorical features (passthrough): {original_categorical_features}")
            logging.debug(f"Model {index+1} (CatBoost): cat_features indices for CatBoost: {cat_feature_indices_for_cb}")


        model = CatBoostClassifier(
            iterations=max_iter,
            depth=6,
            eval_metric='AUC',
            l2_leaf_reg=3,
            learning_rate=0.02,
            loss_function='Logloss',
            logging_level='Silent',
            random_seed=random_state, # CatBoost uses random_seed
            cat_features=cat_feature_indices_for_cb
        )
        pipeline = Pipeline(steps=[('preprocessor', catboost_preprocessor), ('classifier', model)])

    elif model_type == 'knn' or model_type == 'linear':
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.linear_model import LogisticRegression

        # Original preprocessor with OneHotEncoder for KNN and Logistic Regression
        other_models_preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), original_numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), original_categorical_features)
            ]
        )
        if model_type == 'knn':
            model = KNeighborsClassifier(n_neighbors=5)
        elif model_type == 'linear':
            model = LogisticRegression(max_iter=max_iter, random_state=random_state)
        else: # Should not happen given the outer if/elif
            raise ValueError(f"Unexpected model_type '{model_type}' in knn/linear block.")
        pipeline = Pipeline(steps=[('preprocessor', other_models_preprocessor), ('classifier', model)])
    else:
        raise ValueError(f"Invalid model_type provided: {model_type}. Expected 'tree', 'knn', or 'linear'.")

    pipeline.fit(X_resampled, y_resampled)
    accuracy = pipeline.score(X_val, y_val)

    logging.info(
        f"Model {index + 1} ({model_type}) trained. Validation accuracy: {accuracy:.2%}"
    )
    return {'model': pipeline, 'accuracy': accuracy}


def optuna_tuner(X: pd.DataFrame, y: pd.Series, model_type: str, n_trials: int = 10,
                 balance: bool = True, random_state: int = 42,
                 balance_strategy: BalanceStrategy = "over") -> Dict[str, Any]:
    """Run Optuna tuning and return the best fitted pipeline and score."""
    import optuna
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.pipeline import Pipeline

    if not isinstance(y, pd.Series):
        y = pd.Series(y, name="target")

    stratify = y if y.nunique() > 1 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        train_size=0.7,
        random_state=random_state,
        stratify=stratify,
    )

    X_resampled, y_resampled = _rebalance_training_data(
        X_train=X_train,
        y_train=y_train,
        balance=balance,
        balance_strategy=balance_strategy,
        random_state=random_state,
    )

    numerical_features_opt = X_resampled.select_dtypes(include='number').columns.tolist()
    categorical_features_opt = X_resampled.select_dtypes(exclude='number').columns.tolist()

    def objective(trial: optuna.trial.Trial) -> float:
        preprocessor_obj: ColumnTransformer
        model_obj: Any # Placeholder for model type

        if model_type == 'tree':
            from catboost import CatBoostClassifier
            preprocessor_obj = ColumnTransformer(
                transformers=[('num', StandardScaler(), numerical_features_opt)],
                remainder='passthrough'
            )

            cat_feature_indices_obj = []
            if categorical_features_opt:
                cat_feature_indices_obj = [
                    len(numerical_features_opt) + i
                    for i in range(len(categorical_features_opt))
                ]

            depth = trial.suggest_int("depth", 4, 10)
            learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
            iterations_trial = trial.suggest_int("iterations", 100, 500, step=50)

            model_obj = CatBoostClassifier(
                depth=depth,
                iterations=iterations_trial,
                learning_rate=learning_rate,
                eval_metric='AUC',
                random_seed=random_state,
                logging_level='Silent',
                cat_features=cat_feature_indices_obj
            )
        elif model_type == 'linear' or model_type == 'knn':
            from sklearn.linear_model import LogisticRegression
            from sklearn.neighbors import KNeighborsClassifier
            preprocessor_obj = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_features_opt),
                    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features_opt)
                ]
            )
            if model_type == 'linear':
                c_val = trial.suggest_float("C", 1e-3, 1e3, log=True)
                max_iter_trial = trial.suggest_int("max_iter", 100, 500, step=50)
                model_obj = LogisticRegression(C=c_val, max_iter=max_iter_trial, random_state=random_state)
            elif model_type == 'knn':
                n_neighbors = trial.suggest_int("n_neighbors", 1, 20)
                model_obj = KNeighborsClassifier(n_neighbors=n_neighbors)
            else: # Should not happen
                raise ValueError(f"Unexpected model_type '{model_type}' in optuna objective knn/linear block.")
        else:
            raise ValueError(f"Invalid model_type for optuna objective: {model_type}")

        pipeline_obj = Pipeline(steps=[('preprocessor', preprocessor_obj), ('classifier', model_obj)])
        pipeline_obj.fit(X_resampled, y_resampled)
        score = pipeline_obj.score(X_val, y_val) # y_val is a Series, fine for score
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    best_score = study.best_value

    # Final model fitting with best_params
    final_preprocessor: ColumnTransformer
    final_model_instance: Any

    if model_type == 'tree':
        from catboost import CatBoostClassifier
        final_preprocessor = ColumnTransformer(
            transformers=[('num', StandardScaler(), numerical_features_opt)],
            remainder='passthrough'
        )
        final_cat_feature_indices = []
        if categorical_features_opt:
            final_cat_feature_indices = [
                len(numerical_features_opt) + i
                for i in range(len(categorical_features_opt))
            ]
        final_model_instance = CatBoostClassifier(
            depth=best_params.get("depth", 6),
            iterations=best_params.get("iterations", 100),
            learning_rate=best_params.get("learning_rate", 0.02),
            eval_metric='AUC',
            random_seed=random_state,
            logging_level='Silent',
            cat_features=final_cat_feature_indices
        )
    elif model_type == 'linear' or model_type == 'knn':
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        final_preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features_opt),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features_opt)
            ]
        )
        if model_type == 'linear':
            final_model_instance = LogisticRegression(
                C=best_params.get("C", 1.0),
                max_iter=best_params.get("max_iter", 100),
                random_state=random_state
            )
        elif model_type == 'knn':
            final_model_instance = KNeighborsClassifier(
                n_neighbors=best_params.get("n_neighbors", 5)
            )
        else: # Should not happen
            raise ValueError(f"Unexpected model_type '{model_type}' in optuna final model knn/linear block.")
    else:
        raise ValueError(f"Invalid model_type for final optuna pipeline: {model_type}")

    final_pipeline = Pipeline(steps=[('preprocessor', final_preprocessor), ('classifier', final_model_instance)])
    final_pipeline.fit(X_resampled, y_resampled)

    logging.info(f"[Optuna] Best params for {model_type}: {best_params}, best score: {best_score:.2%}")

    return {'model': final_pipeline, 'accuracy': best_score, 'best_params': best_params}
