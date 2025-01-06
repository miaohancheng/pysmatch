# -*- coding: utf-8 -*-
import logging
import numpy as np


def fit_model(index: int, X, y, model_type: str, balance: bool,
              max_iter: int = 100, random_state: int = 42):
    """
    Trains a model (logistic regression / tree / knn) for a given index.
    中文注释: 训练指定类型的模型
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

    numerical_features = X_resampled.select_dtypes(include=np.number).columns
    categorical_features = X_resampled.select_dtypes(exclude=np.number).columns

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
        raise ValueError("Invalid model_type...")

    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    pipeline.fit(X_resampled, y_resampled.iloc[:, 0] if y_resampled.ndim == 2 else y_resampled)
    accuracy = pipeline.score(X_resampled, y_resampled)

    logging.info(f"Model {index + 1} trained. Accuracy: {accuracy:.2%}")
    return {'model': pipeline, 'accuracy': accuracy}


def optuna_tuner(X, y, model_type: str, n_trials: int = 10, balance: bool = True, random_state: int = 42):
    """
    Use optuna to search for best hyperparams for a given model_type.
    中文注释: 使用 optuna 对指定模型类型进行超参数搜索
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

    numerical_features = X_resampled.select_dtypes(include=np.number).columns
    categorical_features = X_resampled.select_dtypes(exclude=np.number).columns

    def objective(trial: optuna.trial.Trial):
        # Build pipeline each trial
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )

        if model_type == 'linear':
            from sklearn.linear_model import LogisticRegression
            c_val = trial.suggest_float("C", 1e-3, 1e3, log=True)
            model = LogisticRegression(C=c_val, max_iter=trial.suggest_int("max_iter", 100, 500, step=50),
                                       random_state=random_state)
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
            raise ValueError("Invalid model_type for optuna...")

        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        pipeline.fit(X_resampled, y_resampled)
        score = pipeline.score(X_val, y_val)
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    best_score = study.best_value

    # Retrain final pipeline on entire (resampled) training set
    # (You might also choose to retrain on X_train+X_val, etc.)
    # For demonstration, we just do it again with best_params:
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
        raise ValueError("Invalid model_type for final pipeline...")

    final_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', final_model)])
    final_pipeline.fit(X_resampled, y_resampled)
    logging.info(f"[Optuna] Best params: {best_params}, best score: {best_score:.2%}")

    # Return final pipeline + best_score
    return {'model': final_pipeline, 'accuracy': best_score, 'best_params': best_params}