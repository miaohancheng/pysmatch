# matching.py
# -*- coding: utf-8 -*-
import logging
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from typing import Optional, Sequence, Union


def _match_output_columns(data: pd.DataFrame, extra_columns: list[str]) -> list[str]:
    """Build the output schema while preserving original column order."""
    columns = list(data.columns)
    for col in extra_columns:
        if col not in columns:
            columns.append(col)
    return columns


def _ensure_record_id(data: pd.DataFrame) -> pd.DataFrame:
    """Attach a stable record identifier when the caller did not provide one."""
    if 'record_id' in data.columns:
        return data

    data_with_ids = data.copy()
    data_with_ids['record_id'] = data_with_ids.index
    return data_with_ids


def _transformed_scores(values: pd.Series, distance: str) -> np.ndarray:
    """Convert propensity scores to the requested matching distance scale."""

    if distance not in {"probability", "logit"}:
        raise ValueError("distance must be 'probability' or 'logit'.")
    scores = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("scores must be finite probabilities between 0 and 1.")
    if distance == "logit":
        scores = np.clip(scores, 1e-8, 1 - 1e-8)
        scores = np.log(scores / (1 - scores))
    return scores


def _row_key(row: pd.Series, columns: Sequence[str]) -> tuple:
    """Build a hashable exact-matching key with stable missing-value handling."""

    key = []
    for column in columns:
        value = row[column]
        key.append("__nan__" if pd.isna(value) else value)
    return tuple(key)


def _validate_exact_columns(data: pd.DataFrame, exact: Optional[Sequence[str]]) -> list[str]:
    columns = list(exact or [])
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"Exact-matching columns not found in data: {missing}")
    return columns

def perform_match(data: pd.DataFrame, yvar: str, threshold: float = 0.001,
                  nmatches: int = 1, method: str = 'min', replacement: bool = False,
                  random_state: Optional[int] = None,
                  distance: str = "probability",
                  exact: Optional[Sequence[str]] = None,
                  chunk_size: Optional[int] = None) -> pd.DataFrame:
    """
    Perform nearest-neighbor matching using propensity scores.

    For each treated sample, this function searches control samples within
    ``threshold`` score distance and selects up to ``nmatches`` controls.
    Selection can be deterministic (``method="min"``) or random.

    Args:
        data (pd.DataFrame): DataFrame containing both test and control groups,
                             must include the `yvar` column and a 'scores' column
                             with propensity scores.
        yvar (str): The name of the binary column indicating group membership (0 or 1).
        threshold (float, optional): The radius within which to search for neighbors
                                     based on propensity score difference. Defaults to 0.001.
        nmatches (int, optional): The maximum number of control matches to find for
                                  each test unit within the specified radius/threshold.
                                  Defaults to 1.
        method (str, optional): Match selection method. Use ``"min"`` (or the
                                backward-compatible alias ``"nearest"``) for smallest
                                score differences, or ``"random"`` for random sampling.
                                Defaults to ``"min"``.
        replacement (bool, optional): Whether control units can be matched multiple times
                                      (used more than once as a match). Defaults to False.
        random_state (Optional[int], optional): Seed used by ``method="random"``.
                                                Defaults to ``None``.
        distance (str, optional): ``"probability"`` for raw score differences or
                                 ``"logit"`` for logit-propensity calipers.
                                 Defaults to ``"probability"``.
        exact (Optional[Sequence[str]], optional): Covariate columns that must
                                                   have equal values in each pair.
                                                   Defaults to ``None``.
        chunk_size (Optional[int], optional): Number of treated rows processed
                                             per radius-neighbor query. ``None``
                                             preserves the original all-at-once
                                             behavior.

    Returns:
        pd.DataFrame: Matched treated/control rows with ``match_id`` and ``record_id``.
                      Returns an empty DataFrame if no matches are found.

    Raises:
        ValueError: If the 'scores' column is not found in the input `data`.
        ValueError: If an invalid `method` parameter is provided (not 'min' or 'random').
    """
    data = _ensure_record_id(data)

    if yvar not in data.columns:
        raise ValueError(f"Treatment column '{yvar}' not found in data.")
    if not 0 <= threshold:
        raise ValueError("threshold must be non-negative.")
    if 'scores' not in data.columns:
        logging.error("No 'scores' column found. Please run predict_scores() first.")
        raise ValueError("Scores column not found in data.")
    if nmatches < 1:
        raise ValueError("nmatches must be at least 1.")
    if chunk_size is not None and (not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1):
        raise ValueError("chunk_size must be a positive integer or None.")

    exact_columns = _validate_exact_columns(data, exact)
    normalized_method = 'min' if method == 'nearest' else method
    if normalized_method == "optimal":
        if nmatches != 1 or replacement:
            raise ValueError("Optimal matching supports nmatches=1 and replacement=False only.")
        return perform_optimal_match(
            data,
            yvar,
            threshold=threshold,
            distance=distance,
            exact=exact_columns,
        )
    if normalized_method not in {'min', 'random'}:
        raise ValueError("Invalid method parameter, use ('random', 'min', 'optimal')")

    # 对测试组和对照组按倾向分数排序
    test_df = data[data[yvar] == 1].copy().reset_index()
    ctrl_df = data[data[yvar] == 0].copy().reset_index()

    test_df = test_df.assign(_distance_score=_transformed_scores(test_df['scores'], distance))
    ctrl_df = ctrl_df.assign(_distance_score=_transformed_scores(ctrl_df['scores'], distance))
    test_df = test_df.sort_values('_distance_score').reset_index(drop=True)
    ctrl_df = ctrl_df.sort_values('_distance_score').reset_index(drop=True)
    test_scores = test_df[['index', 'scores']]
    ctrl_scores = ctrl_df[['index', 'scores']]

    test_indices = test_scores['index'].values
    test_scores_values = _transformed_scores(test_scores['scores'], distance).reshape(-1, 1)
    ctrl_indices = ctrl_scores['index'].values
    ctrl_scores_values = _transformed_scores(ctrl_scores['scores'], distance).reshape(-1, 1)

    output_columns = _match_output_columns(data, ['match_id', 'record_id'])

    if len(test_scores_values) == 0 or len(ctrl_scores_values) == 0:
        return pd.DataFrame(columns=output_columns)

    effective_n_neighbors = min(max(1, nmatches), len(ctrl_scores_values))
    rng = np.random.default_rng(random_state)
    if exact_columns:
        control_positions = {}
        for position, (_, row) in enumerate(ctrl_df.iterrows()):
            control_positions.setdefault(_row_key(row, exact_columns), []).append(position)
        test_keys = [_row_key(row, exact_columns) for _, row in test_df.iterrows()]
        distances = indices = None
    else:
        nbrs = NearestNeighbors(n_neighbors=effective_n_neighbors, radius=threshold, algorithm='ball_tree')
        nbrs.fit(ctrl_scores_values)
        distances = indices = None

    matched_records = []
    current_match_id = 0
    used_ctrl_indices = set() if not replacement else None

    step = chunk_size or len(test_scores_values)
    for batch_start in range(0, len(test_scores_values), step):
        batch_end = min(batch_start + step, len(test_scores_values))
        if exact_columns:
            batch_distances = batch_indices = None
        else:
            batch_distances, batch_indices = nbrs.radius_neighbors(
                test_scores_values[batch_start:batch_end]
            )
        for local_index in range(batch_end - batch_start):
            i = batch_start + local_index
            if exact_columns:
                neighbors = np.asarray(control_positions.get(test_keys[i], []), dtype=int)
                dists = np.abs(ctrl_scores_values[neighbors, 0] - test_scores_values[i, 0])
                within = dists <= threshold
                neighbors = neighbors[within]
                dists = dists[within]
            else:
                dists, neighbors = batch_distances[local_index], batch_indices[local_index]
            if len(neighbors) == 0:
                continue
            if normalized_method == 'min':
                sorted_order = np.argsort(dists)
                selected = []
                for idx in sorted_order:
                    ctrl_idx = ctrl_indices[neighbors[idx]]
                    if not replacement and ctrl_idx in used_ctrl_indices:
                        continue
                    selected.append(ctrl_idx)
                    if not replacement:
                        used_ctrl_indices.add(ctrl_idx)
                    if len(selected) == nmatches:
                        break
                if selected:
                    for ctrl_idx in selected:
                        matched_records.append({
                            'test_index': test_indices[i],
                            'control_index': ctrl_idx,
                            'match_id': current_match_id
                        })
                    current_match_id += 1
            elif normalized_method == 'random':
                possible = list(neighbors)
                if not replacement:
                    possible = [n for n in possible if ctrl_indices[n] not in used_ctrl_indices]
                if len(possible) == 0:
                    continue
                selected = rng.choice(possible, size=min(nmatches, len(possible)), replace=False)
                for n in selected:
                    ctrl_idx = ctrl_indices[n]
                    matched_records.append({
                        'test_index': test_indices[i],
                        'control_index': ctrl_idx,
                        'match_id': current_match_id
                    })
                    if not replacement:
                        used_ctrl_indices.add(ctrl_idx)
                current_match_id += 1
    if matched_records:
        matched_df = pd.DataFrame(matched_records)
        matched_test = data.loc[matched_df['test_index']].copy()
        matched_ctrl = data.loc[matched_df['control_index']].copy()

        matched_test['match_id'] = matched_df['match_id'].values
        matched_ctrl['match_id'] = matched_df['match_id'].values
        if 'record_id' not in matched_test.columns:
            matched_test['record_id'] = matched_test.index
        if 'record_id' not in matched_ctrl.columns:
            matched_ctrl['record_id'] = matched_ctrl.index

        output_df = pd.concat([matched_test, matched_ctrl], ignore_index=True)
    else:
        output_df = pd.DataFrame(columns=output_columns)

    return output_df


def perform_optimal_match(
    data: pd.DataFrame,
    yvar: str,
    threshold: float = 0.001,
    distance: str = "probability",
    exact: Optional[Sequence[str]] = None,
    max_matrix_size: int = 5_000_000,
) -> pd.DataFrame:
    """Perform one-to-one minimum-cost matching with optional calipers.

    The assignment minimizes the total propensity-score distance. Controls
    cannot be reused, ``nmatches`` is implicitly one, and pairs outside the
    caliper are discarded after assignment. This method is intended for
    moderate data sizes because the cost matrix is quadratic.
    """

    data = _ensure_record_id(data)
    if yvar not in data.columns:
        raise ValueError(f"Treatment column '{yvar}' not found in data.")
    if 'scores' not in data.columns:
        raise ValueError("Scores column not found in data.")
    if not 0 <= threshold:
        raise ValueError("threshold must be non-negative.")
    exact_columns = _validate_exact_columns(data, exact)
    test_df = data[data[yvar] == 1].copy()
    control_df = data[data[yvar] == 0].copy()
    output_columns = _match_output_columns(data, ['match_id', 'pair_score_diff'])
    if test_df.empty or control_df.empty:
        return pd.DataFrame(columns=output_columns)
    if len(test_df) * len(control_df) > max_matrix_size:
        raise ValueError(
            "Optimal matching cost matrix is too large; use standard/exhaustive matching "
            "or increase max_matrix_size explicitly."
        )

    from scipy.optimize import linear_sum_assignment

    test_scores = _transformed_scores(test_df['scores'], distance)
    control_scores = _transformed_scores(control_df['scores'], distance)
    costs = np.abs(test_scores[:, None] - control_scores[None, :])
    if exact_columns:
        test_keys = [_row_key(row, exact_columns) for _, row in test_df.iterrows()]
        control_keys = [_row_key(row, exact_columns) for _, row in control_df.iterrows()]
        exact_mask = np.asarray(
            [[test_key == control_key for control_key in control_keys] for test_key in test_keys],
            dtype=bool,
        )
        costs = np.where(exact_mask, costs, np.inf)

    finite_costs = costs[np.isfinite(costs)]
    if finite_costs.size == 0:
        return pd.DataFrame(columns=output_columns)
    large_cost = max(float(finite_costs.max()), float(threshold), 1.0) * 10
    assignment_costs = np.where(np.isfinite(costs), costs, large_cost)
    row_indices, control_indices = linear_sum_assignment(assignment_costs)
    selected = [
        (row, control, float(costs[row, control]))
        for row, control in zip(row_indices, control_indices)
        if np.isfinite(costs[row, control]) and costs[row, control] <= threshold
    ]
    if not selected:
        return pd.DataFrame(columns=output_columns)

    matched_tests = []
    matched_controls = []
    for match_id, (test_position, control_position, pair_distance) in enumerate(selected):
        test_row = test_df.iloc[test_position].drop(labels=['_distance_score'], errors='ignore').copy()
        control_row = control_df.iloc[control_position].drop(labels=['_distance_score'], errors='ignore').copy()
        test_row['match_id'] = match_id
        control_row['match_id'] = match_id
        test_row['pair_score_diff'] = pair_distance
        control_row['pair_score_diff'] = pair_distance
        matched_tests.append(test_row)
        matched_controls.append(control_row)
    return pd.concat(matched_tests + matched_controls, axis=1).T.reset_index(drop=True)


def perform_exhaustive_match(
    data: pd.DataFrame,
    yvar: str,
    threshold: float = 0.001,
    nmatches: int = 1,
    show_progress: bool = False,
    distance: str = "probability",
    exact: Optional[Sequence[str]] = None,
    chunk_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    Perform exhaustive matching while prioritizing unused controls first.

    Controls are pre-sorted by propensity score once, then candidate windows are
    found by binary search for each treated row. Within each window, selection is
    ordered by:
    1. whether the control has been used before,
    2. current usage count,
    3. absolute score distance.
    """
    data = _ensure_record_id(data)

    if yvar not in data.columns:
        raise ValueError(f"Treatment column '{yvar}' not found in data.")
    if not 0 <= threshold:
        raise ValueError("threshold must be non-negative.")
    if 'scores' not in data.columns:
        logging.error("No 'scores' column found. Please run predict_scores() first.")
        raise ValueError("Scores column not found in data.")
    if nmatches < 1:
        raise ValueError("nmatches must be at least 1.")
    if chunk_size is not None and (not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1):
        raise ValueError("chunk_size must be a positive integer or None.")

    exact_columns = _validate_exact_columns(data, exact)
    output_columns = _match_output_columns(data, ['match_id', 'matched_as', 'pair_score_diff'])
    test_df = data[data[yvar] == 1].copy()
    control_df = data[data[yvar] == 0].copy()

    if test_df.empty or control_df.empty:
        return pd.DataFrame(columns=output_columns)

    control_df = control_df.copy()
    control_df['_distance_score'] = _transformed_scores(control_df['scores'], distance)
    control_df = control_df.sort_values('_distance_score', kind='mergesort').reset_index(drop=True)
    control_scores = control_df['_distance_score'].to_numpy()
    control_record_ids = control_df['record_id'].to_numpy()
    control_usage_counts = np.zeros(len(control_df), dtype=int)

    test_df = test_df.copy()
    test_df['_distance_score'] = _transformed_scores(test_df['scores'], distance)

    pairs: list[dict[str, object]] = []
    test_key_by_id = {}
    if exact_columns:
        test_key_by_id = {
            row['record_id']: _row_key(row, exact_columns)
            for _, row in test_df.iterrows()
        }

    exact_positions = {}
    if exact_columns:
        for position, (_, row) in enumerate(control_df.iterrows()):
            exact_positions.setdefault(_row_key(row, exact_columns), []).append(position)

    step = chunk_size or len(test_df)
    for batch_start in range(0, len(test_df), step):
        batch = test_df.iloc[batch_start:min(batch_start + step, len(test_df))]
        iterator = batch[['record_id', '_distance_score']].itertuples(index=False, name=None)
        if show_progress and chunk_size is None:
            from tqdm import tqdm

            iterator = tqdm(iterator, total=len(test_df), desc="Exhaustive Matching")
        for case_record_id, case_score in iterator:
            if exact_columns:
                positions = np.asarray(exact_positions.get(test_key_by_id[case_record_id], []), dtype=int)
                candidate_scores = control_scores[positions]
                left = np.searchsorted(candidate_scores, case_score - threshold, side='left')
                right = np.searchsorted(candidate_scores, case_score + threshold, side='right')
                candidate_positions = positions[left:right]
            else:
                left = np.searchsorted(control_scores, case_score - threshold, side='left')
                right = np.searchsorted(control_scores, case_score + threshold, side='right')
                candidate_positions = np.arange(left, right)

            if len(candidate_positions) == 0:
                continue

            candidate_usage = control_usage_counts[candidate_positions]
            candidate_diffs = np.abs(control_scores[candidate_positions] - case_score)
            candidate_is_used = candidate_usage > 0
            order = np.lexsort(
                (candidate_positions, candidate_diffs, candidate_usage, candidate_is_used)
            )
            selected_positions = candidate_positions[order[:nmatches]]

            for pos in selected_positions:
                pairs.append(
                    {
                        'case_record_id': case_record_id,
                        'control_record_id': control_record_ids[pos],
                        'pair_score_diff': float(abs(control_scores[pos] - case_score)),
                    }
                )
            control_usage_counts[selected_positions] += 1

    if not pairs:
        return pd.DataFrame(columns=output_columns)

    pairs_df = pd.DataFrame(pairs)
    pairs_df['match_id'] = np.arange(len(pairs_df))

    case_rows = pairs_df[['case_record_id', 'match_id', 'pair_score_diff']].rename(
        columns={'case_record_id': 'record_id'}
    )
    case_rows['matched_as'] = 'case'
    case_rows['_pair_order'] = pairs_df['match_id'] * 2

    control_rows = pairs_df[['control_record_id', 'match_id', 'pair_score_diff']].rename(
        columns={'control_record_id': 'record_id'}
    )
    control_rows['matched_as'] = 'control'
    control_rows['_pair_order'] = pairs_df['match_id'] * 2 + 1

    annotation = pd.concat([case_rows, control_rows], ignore_index=True)
    annotation = annotation.sort_values('_pair_order', kind='mergesort').drop(columns=['_pair_order'])

    row_lookup = data.drop_duplicates(subset='record_id', keep='first')
    matched_data = annotation.merge(row_lookup, on='record_id', how='left', sort=False)
    matched_data = matched_data[list(data.columns) + ['match_id', 'matched_as', 'pair_score_diff']]

    return matched_data


def tune_threshold(data: pd.DataFrame, yvar: str, method: str = 'min',
                   nmatches: int = 1, rng: Union[np.ndarray, None] = None,
                   random_state: Optional[int] = None,
                   distance: str = "probability",
                   exact: Optional[Sequence[str]] = None,
                   chunk_size: Optional[int] = None) -> tuple:
    """
    Evaluates matching retention across a range of threshold values.

    Performs matching using `perform_match` for each threshold in the specified
    range (`rng`) and calculates the proportion of the original minority group
    that is retained in the matched dataset. Useful for choosing a threshold.

    Args:
        data (pd.DataFrame): The input DataFrame containing scores and `yvar`.
        yvar (str): The name of the binary treatment/control indicator column.
        method (str, optional): The matching method ('min' or 'random') to use for
                                each evaluation. Defaults to 'min'.
        nmatches (int, optional): The number of matches to seek for each test unit.
                                  Defaults to 1.
        rng (Optional[np.ndarray], optional): A NumPy array of threshold values to test.
                                              If None, defaults to `np.arange(0, 0.001, 0.0001)`.
                                              Defaults to None.
        random_state (Optional[int], optional): Seed used when ``method="random"``.
                                                Defaults to ``None``.
        distance (str, optional): Matching distance scale, ``"probability"`` or
                                 ``"logit"``. Defaults to ``"probability"``.
        exact (Optional[Sequence[str]], optional): Exact-matching columns.
        chunk_size (Optional[int], optional): Treated-row chunk size passed to
                                              ``perform_match``.

    Returns:
        Tuple[np.ndarray, list]: A tuple containing:
            - thresholds (np.ndarray): The array of threshold values tested.
            - retained (list): A list of proportions (float) of the minority group
                               retained for each corresponding threshold.
    """
    if rng is None:
        rng = np.arange(0, 0.001, 0.0001)
    thresholds = []
    retained = []
    for threshold in rng:
        matched_data = perform_match(data, yvar, threshold=threshold,
                                     nmatches=nmatches, method=method, replacement=False,
                                     random_state=random_state, distance=distance, exact=exact,
                                     chunk_size=chunk_size)
        prop = prop_retained(data, matched_data, yvar)
        thresholds.append(threshold)
        retained.append(prop)
    return thresholds, retained


def prop_retained(original_data: pd.DataFrame, matched_data: pd.DataFrame, yvar: str) -> float:
    """
    Calculates the proportion of the minority group retained after matching.

    Compares the number of unique minority group members in the matched dataset
    to the number in the original dataset.

    Args:
        original_data (pd.DataFrame): The dataset before matching.
        matched_data (pd.DataFrame): The dataset after matching. Should contain 'record_id'
                                     or rely on index if 'record_id' is missing.
        yvar (str): The name of the binary treatment/control indicator column.

    Returns:
        float: The proportion (0.0 to 1.0) of the original minority group present
               in the matched dataset. Returns 0.0 if the original minority group
               was empty.
    """
    minority = 1 if (original_data[yvar] == 1).sum() <= (original_data[yvar] == 0).sum() else 0
    original_minority = original_data[original_data[yvar] == minority]
    matched_minority = matched_data[matched_data[yvar] == minority]

    original_id_col = "record_id" if "record_id" in original_minority.columns else None
    matched_id_col = "record_id" if "record_id" in matched_minority.columns else None

    if original_id_col is not None:
        denom = original_minority[original_id_col].nunique()
    else:
        denom = original_minority.index.nunique()

    if matched_id_col is not None:
        num = matched_minority[matched_id_col].nunique()
    else:
        num = matched_minority.index.nunique()

    return num / denom if denom > 0 else 0.0
