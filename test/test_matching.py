import numpy as np
import pandas as pd
import pytest

from pysmatch.matching import perform_match, tune_threshold, prop_retained


@pytest.fixture
def sample_match_data():
    return pd.DataFrame(
        {
            "record_id": [0, 1, 2, 3, 4, 5],
            "treated": [1, 1, 0, 0, 0, 0],
            "scores": [0.10, 0.80, 0.11, 0.12, 0.78, 0.81],
            "x": [1, 2, 3, 4, 5, 6],
        }
    )


def test_perform_match_returns_expected_columns(sample_match_data):
    matched = perform_match(
        data=sample_match_data,
        yvar="treated",
        threshold=0.03,
        nmatches=1,
        method="min",
        replacement=False,
    )
    assert not matched.empty
    assert "match_id" in matched.columns
    assert "record_id" in matched.columns
    assert set(matched["treated"].unique()) == {0, 1}


def test_perform_match_requires_scores_column():
    data = pd.DataFrame({"treated": [1, 0], "x": [1, 2]})
    with pytest.raises(ValueError, match="Scores column not found"):
        perform_match(data=data, yvar="treated")


def test_tune_threshold_outputs_probabilities(sample_match_data):
    thresholds, retained = tune_threshold(
        data=sample_match_data,
        yvar="treated",
        method="min",
        nmatches=1,
        rng=np.array([0.01, 0.03, 0.10]),
    )
    assert len(thresholds) == 3
    assert len(retained) == 3
    assert all(0.0 <= r <= 1.0 for r in retained)


def test_prop_retained_uses_unique_ids():
    original = pd.DataFrame(
        {
            "record_id": [10, 11, 20, 21],
            "treated": [1, 1, 0, 0],
            "scores": [0.2, 0.8, 0.21, 0.79],
        }
    )
    matched = pd.DataFrame(
        {
            "record_id": [10, 10, 11, 20, 21],
            "treated": [1, 1, 1, 0, 0],
            "scores": [0.2, 0.2, 0.8, 0.21, 0.79],
            "match_id": [0, 1, 1, 0, 1],
        }
    )
    retained = prop_retained(original, matched, "treated")
    assert retained == 1.0
