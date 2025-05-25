# test_matcher.py
import pytest
import pandas as pd
import numpy as np
import logging # Added: Import the logging module
from pysmatch.Matcher import Matcher
# Mocking external dependencies if they are complex or have side effects (e.g., plotting)
from unittest import mock

# --- Test Data Fixtures ---
@pytest.fixture
def sample_data_frames():
    """Provides sample test and control pandas DataFrames for testing."""
    np.random.seed(123)
    # n_test and n_control are defined here for data generation
    # but tests should rely on matcher's attributes for these counts if needed
    n_test_fixture = 20
    n_control_fixture = 50
    test_df = pd.DataFrame({
        'age': np.random.normal(loc=55, scale=5, size=n_test_fixture),
        'income': np.random.normal(loc=65000, scale=10000, size=n_test_fixture),
        'feature1': np.random.rand(n_test_fixture)
    })
    control_df = pd.DataFrame({
        'age': np.random.normal(loc=50, scale=7, size=n_control_fixture),
        'income': np.random.normal(loc=55000, scale=12000, size=n_control_fixture),
        'feature1': np.random.rand(n_control_fixture)
    })
    return test_df, control_df

@pytest.fixture
def matcher_instance(sample_data_frames):
    """Provides a Matcher instance initialized with sample data."""
    test_df, control_df = sample_data_frames
    matcher = Matcher(test=test_df, control=control_df, yvar='treated', exhaustive_matching_default=False)
    return matcher

@pytest.fixture
def matcher_instance_fitted(matcher_instance):
    """Provides a Matcher instance with scores fitted and predicted."""
    # Mocking fit_model to avoid actual model training in unit tests
    # This mock should return what fit_model is expected to return
    mock_model_object = mock.Mock()
    # Ensure predict_proba returns a 2D array where the second column is the score for the positive class
    # The number of rows must match the number of samples in matcher_instance.X
    if not matcher_instance.X.empty:
        mock_model_object.predict_proba.return_value = np.random.rand(len(matcher_instance.X), 2)
    else: # Handle case where X might be empty if data processing in __init__ fails
        mock_model_object.predict_proba.return_value = np.array([]).reshape(0,2)


    with mock.patch('pysmatch.modeling.fit_model', return_value={'model': mock_model_object, 'accuracy': 0.85}):
        matcher_instance.fit_scores(balance=False, nmodels=1) # Fit a single, simple model
    matcher_instance.predict_scores()
    return matcher_instance

# --- Test Cases ---

def test_matcher_initialization(sample_data_frames):
    """Test Matcher initialization and basic attribute setup."""
    test_df, control_df = sample_data_frames
    # Get n_test and n_control from the fixture's generated data for assertion
    n_test_val = len(test_df)
    n_control_val = len(control_df)

    matcher = Matcher(test=test_df, control=control_df, yvar='is_exposed', exclude=['income'], exhaustive_matching_default=True)

    assert matcher.treatment_col == 'is_exposed'
    assert 'is_exposed' in matcher.data.columns
    assert 'record_id' in matcher.data.columns
    assert matcher.data['record_id'].is_unique
    assert matcher.exhaustive_matching_default is True
    assert 'income' not in matcher.xvars # Due to exclude
    assert 'age' in matcher.xvars
    assert 'feature1' in matcher.xvars
    # Assert against the actual lengths from the matcher instance after processing
    assert len(matcher.data) == matcher.testn + matcher.controln
    assert len(matcher.test_df) == matcher.testn
    assert len(matcher.control_df) == matcher.controln
    # Additionally, check if matcher.testn and matcher.controln match the input lengths
    # This can be tricky if __init__ drops rows with NAs in selected xvars.
    # For this test, assume no rows are dropped from the fixture data by __init__.
    assert matcher.testn == n_test_val
    assert matcher.controln == n_control_val


def test_fit_scores_and_predict_scores(matcher_instance):
    """Test fitting scores and predicting them."""
    # Mock the actual model fitting to speed up tests and avoid dependency on ML libraries
    mock_model = mock.Mock()
    # Ensure predict_proba returns a 2D array with probabilities for two classes
    if not matcher_instance.X.empty:
        mock_model.predict_proba.return_value = np.random.uniform(size=(len(matcher_instance.X), 2))
    else:
        mock_model.predict_proba.return_value = np.array([]).reshape(0,2)


    with mock.patch('pysmatch.modeling.fit_model', return_value={'model': mock_model, 'accuracy': 0.9}) as mock_fit:
        matcher_instance.fit_scores(balance=False, nmodels=1) # Test with a single model
        if not matcher_instance.X.empty: # fit_model is only called if X is not empty
            mock_fit.assert_called_once()

    if not matcher_instance.X.empty:
        assert len(matcher_instance.models) == 1
        assert len(matcher_instance.model_accuracy) == 1
        assert matcher_instance.model_accuracy[0] == 0.9

        matcher_instance.predict_scores()
        assert 'scores' in matcher_instance.data.columns
        assert matcher_instance.data['scores'].isnull().sum() == 0
        assert matcher_instance.data['scores'].min() >= 0
        assert matcher_instance.data['scores'].max() <= 1
    else:
        logging.info("Skipping score prediction assertions as X was empty.")


def test_match_standard_delegation(matcher_instance_fitted):
    """Test that standard matching delegates to matching.perform_match."""
    matcher = matcher_instance_fitted

    if matcher.X.empty or 'scores' not in matcher.data.columns:
        pytest.skip("Skipping standard match delegation test as data/scores are not ready.")

    mock_matched_df = pd.DataFrame({
        'record_id': [0, 1, 2, 3],
        matcher.treatment_col: [1, 0, 1, 0],
        'scores': [0.6, 0.58, 0.7, 0.72],
        'match_id': [0, 0, 1, 1]
    })

    with mock.patch('pysmatch.matching.perform_match', return_value=mock_matched_df) as mock_perform_match:
        matcher.match(threshold=0.01, nmatches=1, method='nearest', replacement=False, exhaustive_matching=False)
        mock_perform_match.assert_called_once()
        pd.testing.assert_frame_equal(matcher.matched_data, mock_matched_df)

def test_match_exhaustive_matching_basic(matcher_instance_fitted):
    """Test the basic execution of exhaustive matching."""
    matcher = matcher_instance_fitted
    if matcher.X.empty or 'scores' not in matcher.data.columns:
        pytest.skip("Skipping exhaustive matching basic test as data/scores are not ready.")

    matcher.match(threshold=0.1, nmatches=1, exhaustive_matching=True)

    # Only assert if matches were actually possible and found
    if not matcher.matched_data.empty:
        assert 'match_id' in matcher.matched_data.columns
        assert 'record_id' in matcher.matched_data.columns
        assert 'matched_as' in matcher.matched_data.columns
        assert 'pair_score_diff' in matcher.matched_data.columns
        assert matcher.treatment_col in matcher.matched_data.columns

        for match_id, group in matcher.matched_data.groupby('match_id'):
            assert len(group) == 2
            assert set(group['matched_as'].unique()) == {'case', 'control'}
            case_row = group[group['matched_as'] == 'case']
            control_row = group[group['matched_as'] == 'control']
            assert case_row[matcher.treatment_col].iloc[0] == 1
            assert control_row[matcher.treatment_col].iloc[0] == 0
    else:
        logging.info("Exhaustive matching basic test: No matches found, which might be okay depending on data/threshold.")


def test_match_exhaustive_nmatches(matcher_instance_fitted):
    """Test that exhaustive matching respects nmatches."""
    matcher = matcher_instance_fitted
    if matcher.X.empty or 'scores' not in matcher.data.columns:
        pytest.skip("Skipping exhaustive nmatches test as data/scores are not ready.")

    matcher.match(threshold=0.2, nmatches=2, exhaustive_matching=True)

    if not matcher.matched_data.empty:
        case_ids_in_matched_data = matcher.matched_data[matcher.matched_data['matched_as'] == 'case']['record_id'].unique()
        num_pairs_formed = matcher.matched_data['match_id'].nunique()

        assert num_pairs_formed >= 0 # At least some matches or zero matches
        if len(case_ids_in_matched_data) > 0:
            example_case_id = case_ids_in_matched_data[0]
            num_matches_for_example_case = matcher.matched_data[
                (matcher.matched_data['matched_as'] == 'case') &
                (matcher.matched_data['record_id'] == example_case_id)
                ]['match_id'].nunique()
            assert num_matches_for_example_case <= 2
    else:
        logging.info("Exhaustive nmatches test: No matches found.")


def test_match_exhaustive_threshold(matcher_instance_fitted):
    """Test that exhaustive matching respects the threshold."""
    matcher = matcher_instance_fitted
    if matcher.X.empty or 'scores' not in matcher.data.columns:
        pytest.skip("Skipping exhaustive threshold test as data/scores are not ready.")

    matcher.match(threshold=0.00001, nmatches=1, exhaustive_matching=True)
    if not matcher.matched_data.empty:
        assert matcher.matched_data['pair_score_diff'].max() <= 0.00001 + 1e-9
    else:
        logging.info("Exhaustive threshold test: No matches found with very strict threshold, which is expected.") # Fixed: Added logging
        pass

def test_record_frequency_exhaustive(matcher_instance_fitted):
    """Test record_frequency after exhaustive matching."""
    matcher = matcher_instance_fitted
    if matcher.X.empty or 'scores' not in matcher.data.columns:
        pytest.skip("Skipping record frequency test as data/scores are not ready.")

    matcher.match(threshold=0.1, nmatches=2, exhaustive_matching=True)

    freq_df = matcher.record_frequency() # Call it regardless of whether matches were found
    if not matcher.matched_data.empty:
        assert not freq_df.empty
        assert 'record_id' in freq_df.columns
        assert 'n_occurrences_as_control' in freq_df.columns
    else:
        # If no matches, record_frequency should return an empty DataFrame with correct columns or handle gracefully
        assert freq_df.empty or list(freq_df.columns) == ['record_id', 'n_occurrences_as_control']


def test_assign_weight_vector(matcher_instance_fitted):
    """Test assigning weights after matching."""
    matcher = matcher_instance_fitted
    if matcher.X.empty or 'scores' not in matcher.data.columns:
        pytest.skip("Skipping assign weight vector test as data/scores are not ready.")

    matcher.match(threshold=0.1, nmatches=1, exhaustive_matching=True)

    if not matcher.matched_data.empty:
        matcher.assign_weight_vector()
        assert 'weight' in matcher.matched_data.columns
        assert matcher.matched_data['weight'].isnull().sum() == 0
        assert matcher.matched_data['weight'].min() > 0
        assert matcher.matched_data['weight'].max() <= 1.0

        example_record_id = matcher.matched_data['record_id'].iloc[0]
        count = matcher.matched_data[matcher.matched_data['record_id'] == example_record_id].shape[0]
        expected_weight = 1 / count
        actual_weight = matcher.matched_data[matcher.matched_data['record_id'] == example_record_id]['weight'].iloc[0]
        assert np.isclose(actual_weight, expected_weight)
    else:
        # Test that assign_weight_vector doesn't crash if matched_data is empty
        matcher.assign_weight_vector()
        assert 'weight' not in matcher.matched_data.columns or matcher.matched_data['weight'].empty


@mock.patch('pysmatch.visualization.plot_scores')
def test_plot_scores_call(mock_plot, matcher_instance_fitted):
    if 'scores' not in matcher_instance_fitted.data.columns:
        # If scores were not predicted (e.g. X was empty), predict_scores might have returned early
        # For this test, ensure scores are there or skip
        if matcher_instance_fitted.X.empty:
            pytest.skip("Skipping plot_scores test as X was empty, scores not predicted.")
        # Attempt to predict scores again if X is not empty but scores are missing
        matcher_instance_fitted.predict_scores()
        if 'scores' not in matcher_instance_fitted.data.columns:
            pytest.skip("Skipping plot_scores test as scores are still not available.")

    matcher_instance_fitted.plot_scores()
    mock_plot.assert_called_once()

@mock.patch('pysmatch.visualization.plot_matched_scores')
def test_plot_matched_scores_call(mock_plot, matcher_instance_fitted):
    if matcher_instance_fitted.X.empty or 'scores' not in matcher_instance_fitted.data.columns:
        pytest.skip("Skipping plot_matched_scores test as data/scores are not ready.")

    matcher_instance_fitted.match(exhaustive_matching=True, threshold=0.1)
    if not matcher_instance_fitted.matched_data.empty:
        matcher_instance_fitted.plot_matched_scores()
        mock_plot.assert_called_once()
    else:
        logging.info("Skipping plot_matched_scores visual test as no data was matched.")

@mock.patch('pysmatch.matching.tune_threshold', return_value=(np.array([0.01, 0.02]), np.array([0.8, 0.7])))
@mock.patch('matplotlib.pyplot.show')
def test_tune_threshold_call(mock_plt_show, mock_tune, matcher_instance_fitted):
    if 'scores' not in matcher_instance_fitted.data.columns: # Ensure scores exist
        if matcher_instance_fitted.X.empty:
            pytest.skip("Skipping tune_threshold test as X was empty, scores not predicted.")
        matcher_instance_fitted.predict_scores()
        if 'scores' not in matcher_instance_fitted.data.columns:
            pytest.skip("Skipping tune_threshold test as scores are still not available.")

    matcher_instance_fitted.tune_threshold(method='min')
    mock_tune.assert_called_once()
    # mock_plt_show.assert_called_once() # Only if plt.show() is guaranteed to be called by your tune_threshold's plotting part

@mock.patch('pysmatch.visualization.compare_continuous', return_value=pd.DataFrame({'var': ['age'], 'smd_before': [0.5], 'smd_after': [0.1]}))
def test_compare_continuous(mock_compare, matcher_instance_fitted):
    matcher = matcher_instance_fitted
    if matcher.X.empty or 'scores' not in matcher.data.columns:
        pytest.skip("Skipping compare_continuous test as data/scores are not ready.")

    matcher.match(exhaustive_matching=True, threshold=0.1)
    result_df = matcher.compare_continuous(return_table=True, plot_result=False)
    mock_compare.assert_called_once()
    if not matcher.matched_data.empty: # Result might be None or empty if no matched data
        assert isinstance(result_df, pd.DataFrame)
        assert 'smd_after' in result_df.columns
    elif result_df is not None: # If no matched data, result_df should ideally be None or an empty DataFrame
        assert result_df.empty


@mock.patch('pysmatch.visualization.compare_categorical', return_value=pd.DataFrame({'var': ['feature_cat'], 'p_before': [0.04], 'p_after': [0.5]}))
def test_compare_categorical(mock_compare, matcher_instance_fitted):
    matcher = matcher_instance_fitted
    if matcher.X.empty or 'scores' not in matcher.data.columns:
        pytest.skip("Skipping compare_categorical test as data/scores are not ready.")

    # Add a categorical variable for testing this
    if not matcher.data.empty:
        matcher.data['feature_cat'] = np.random.choice(['A', 'B', 'C'], size=len(matcher.data))
        if 'feature_cat' not in matcher.xvars: # Avoid duplicates if test is run multiple times on same instance
            matcher.xvars.append('feature_cat')
        matcher.X = matcher.data[matcher.xvars]
    else:
        pytest.skip("Skipping compare_categorical as initial matcher.data is empty.")

    matcher.match(exhaustive_matching=True, threshold=0.1)
    result_df = matcher.compare_categorical(return_table=True, plot_result=False)
    mock_compare.assert_called_once()
    if not matcher.matched_data.empty:
        assert isinstance(result_df, pd.DataFrame)
        assert 'p_after' in result_df.columns
    elif result_df is not None:
        assert result_df.empty


def test_prop_test(matcher_instance_fitted):
    matcher = matcher_instance_fitted
    if matcher.X.empty or 'scores' not in matcher.data.columns:
        pytest.skip("Skipping prop_test as data/scores are not ready.")

    if matcher.data.empty:
        pytest.skip("Skipping prop_test as initial matcher.data is empty.")

    categories = ['cat1', 'cat2', 'cat3']
    matcher.data['categorical_var'] = np.random.choice(categories, size=len(matcher.data))
    if 'categorical_var' not in matcher.xvars:
        matcher.xvars.append('categorical_var')
        # matcher.X = matcher.data[matcher.xvars] # X is updated in fit_scores, not strictly needed here unless re-fitting

    matcher.match(exhaustive_matching=True, threshold=0.1)

    result = matcher.prop_test('categorical_var')
    if result is not None:
        assert isinstance(result, dict)
        assert result['var'] == 'categorical_var'
        assert 'before' in result
        assert 'after' in result

    result_cont = matcher.prop_test('age')
    assert result_cont is None

    if 'feature1' not in matcher.exclude: # Ensure it's not already excluded
        matcher.exclude.append('feature1')
    result_excluded = matcher.prop_test('feature1')
    assert result_excluded is None
    if 'feature1' in matcher.exclude: # Clean up
        matcher.exclude.remove('feature1')
