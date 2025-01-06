# -*- coding: utf-8 -*-
import logging
import numpy as np
import pandas as pd


def perform_match(data, yvar, threshold=0.001, nmatches=1, method='min', replacement=False):
    """
    Finds suitable match(es) for each record in the minority dataset, if one exists.
    中文注释: 给少数类匹配相似记录
    """
    from sklearn.neighbors import NearestNeighbors

    if 'scores' not in data.columns:
        logging.info("No 'scores' column found, please run predict_scores() first.")
        return pd.DataFrame()

    test_scores = data[data[yvar] == 1][['scores']].sort_values('scores').reset_index()
    ctrl_scores = data[data[yvar] == 0][['scores']].sort_values('scores').reset_index()

    test_indices = test_scores['index'].values
    test_scores_values = test_scores['scores'].values.reshape(-1, 1)
    ctrl_indices = ctrl_scores['index'].values
    ctrl_scores_values = ctrl_scores['scores'].values.reshape(-1, 1)

    nbrs = NearestNeighbors(n_neighbors=nmatches, radius=threshold, algorithm='ball_tree').fit(ctrl_scores_values)
    distances, indices = nbrs.radius_neighbors(test_scores_values)

    matched_records = []
    current_match_id = 0
    used_ctrl_indices = set() if not replacement else None

    for i, neighbors in enumerate(indices):
        if len(neighbors) == 0:
            continue
        if method == 'min':
            sorted_neighbors = neighbors[np.argsort(distances[i])]
            selected = []
            for neighbor in sorted_neighbors:
                ctrl_idx = ctrl_indices[neighbor]
                if (not replacement) and (ctrl_idx in used_ctrl_indices):
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
            select = min(nmatches, len(possible))
            selected = np.random.choice(possible, size=select, replace=False).tolist()
            for neighbor in selected:
                ctrl_idx = ctrl_indices[neighbor]
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
        output_df = pd.DataFrame(columns=data.columns.tolist() + ['match_id', 'record_id'])

    return output_df


def tune_threshold(data, yvar, method='min', nmatches=1, rng=None):
    """
    Matches data over a grid to optimize threshold value and returns the proportion retained.
    中文注释: 在给定阈值网格上做匹配，查看保留率
    """
    if rng is None:
        rng = np.arange(0, .001, .0001)

    thresholds = []
    retained = []
    for i in rng:
        matched_data = perform_match(data, yvar, threshold=i, nmatches=nmatches, method=method, replacement=False)
        prop = prop_retained(data, matched_data, yvar)
        thresholds.append(i)
        retained.append(prop)
    return thresholds, retained


def prop_retained(original_data, matched_data, yvar):
    """
    Returns the proportion of data retained after matching
    中文注释: 返回匹配后少数类样本保留的比例
    """
    minority = 1 if sum(original_data[yvar] == 1) <= sum(original_data[yvar] == 0) else 0
    denom = len(original_data[original_data[yvar] == minority])
    num = len(matched_data[matched_data[yvar] == minority])
    return num * 1.0 / denom if denom > 0 else 0