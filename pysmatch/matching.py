# matching.py
# -*- coding: utf-8 -*-
import logging
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from typing import Union

def perform_match(data: pd.DataFrame, yvar: str, threshold: float = 0.001,
                  nmatches: int = 1, method: str = 'min', replacement: bool = False) -> pd.DataFrame:
    """
    Performs nearest neighbor matching based on propensity scores within a radius.

    Finds suitable match(es) from the control group for each record in the test
    (treatment) group based on propensity scores ('scores' column). It uses
    `sklearn.neighbors.NearestNeighbors` with `radius=threshold` to find potential
    neighbors and then applies selection logic based on the `method` parameter
    ('min' or 'random') to choose up to `nmatches`.

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
        method (str, optional): The method for selecting matches among neighbors found
                                within the radius. Options:
                                'min': Selects the `nmatches` neighbors with the smallest
                                       score difference.
                                'random': Selects `nmatches` neighbors randomly from those
                                          within the radius.
                                Defaults to 'min'.
        replacement (bool, optional): Whether control units can be matched multiple times
                                      (used more than once as a match). Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing the matched test and control units.
                      Includes original columns plus 'match_id' (linking matched pairs/groups)
                      and 'record_id' (preserving the original index of the unit). Returns
                      an empty DataFrame if no matches are found.

    Raises:
        ValueError: If the 'scores' column is not found in the input `data`.
        ValueError: If an invalid `method` parameter is provided (not 'min' or 'random').
    """
    if 'scores' not in data.columns:
        logging.error("No 'scores' column found. Please run predict_scores() first.")
        raise ValueError("Scores column not found in data.")

    # 对测试组和对照组按倾向分数排序
    test_df = data[data[yvar] == 1].copy().reset_index()
    ctrl_df = data[data[yvar] == 0].copy().reset_index()

    test_scores = test_df[['index', 'scores']].sort_values('scores').reset_index(drop=True)
    ctrl_scores = ctrl_df[['index', 'scores']].sort_values('scores').reset_index(drop=True)

    test_indices = test_scores['index'].values
    test_scores_values = test_scores['scores'].values.reshape(-1, 1)
    ctrl_indices = ctrl_scores['index'].values
    ctrl_scores_values = ctrl_scores['scores'].values.reshape(-1, 1)

    nbrs = NearestNeighbors(n_neighbors=nmatches, radius=threshold, algorithm='ball_tree')
    nbrs.fit(ctrl_scores_values)
    distances, indices = nbrs.radius_neighbors(test_scores_values)

    matched_records = []
    current_match_id = 0
    used_ctrl_indices = set() if not replacement else None

    for i, (dists, neighbors) in enumerate(zip(distances, indices)):
        if len(neighbors) == 0:
            continue
        if method == 'min':
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
        elif method == 'random':
            possible = list(neighbors)
            if not replacement:
                possible = [n for n in possible if ctrl_indices[n] not in used_ctrl_indices]
            if len(possible) == 0:
                continue
            selected = np.random.choice(possible, size=min(nmatches, len(possible)), replace=False)
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
        else:
            raise ValueError("Invalid method parameter, use ('random', 'min')")

    if matched_records:
        matched_df = pd.DataFrame(matched_records)
        matched_test = data.loc[matched_df['test_index']].copy()
        matched_ctrl = data.loc[matched_df['control_index']].copy()

        matched_test['match_id'] = matched_df['match_id'].values
        matched_ctrl['match_id'] = matched_df['match_id'].values
        matched_test['record_id'] = matched_test.index
        matched_ctrl['record_id'] = matched_ctrl.index

        output_df = pd.concat([matched_test, matched_ctrl], ignore_index=True)
    else:
        output_df = pd.DataFrame(columns=list(data.columns) + ['match_id', 'record_id'])

    return output_df


def tune_threshold(data: pd.DataFrame, yvar: str, method: str = 'min',
                   nmatches: int = 1, rng: Union[np.ndarray, None] = None) -> tuple:
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
                                     nmatches=nmatches, method=method, replacement=False)
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
    denom = len(original_data[original_data[yvar] == minority])
    num = len(matched_data[matched_data[yvar] == minority])
    return num / denom if denom > 0 else 0