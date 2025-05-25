# test_matcher.py
import pytest
import pandas as pd
import numpy as np
from pysmatch.Matcher import Matcher
# Mocking external dependencies if they are complex or have side effects (e.g., plotting)
from unittest import mock

# --- Test Data Fixtures ---
@pytest.fixture
def sample_data_frames():
    """Provides sample test and control pandas DataFrames for testing."""
    np.random.seed(123)
    n_test = 20 # Smaller N for faster tests
    n_control = 50
    test_df = pd.DataFrame({
        'age': np.random.normal(loc=55, scale=5, size=n_test),
        'income': np.random.normal(loc=65000, scale=10000, size=n_test),
        'feature1': np.random.rand(n_test)
    })
    control_df = pd.DataFrame({
        'age': np.random.normal(loc=50, scale=7, size=n_control),
        'income': np.random.normal(loc=55000, scale=12000, size=n_control),
        'feature1': np.random.rand(n_control)
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
    mock_model_object.predict_proba.return_value = np.random.rand(len(matcher_instance.X), 2) # Mock predict_proba

    with mock.patch('pysmatch.modeling.fit_model', return_value={'model': mock_model_object, 'accuracy': 0.85}):
        matcher_instance.fit_scores(balance=False, nmodels=1) # Fit a single, simple model
    matcher_instance.predict_scores()
    return matcher_instance

# --- Test Cases ---

def test_matcher_initialization(sample_data_frames):
    """Test Matcher initialization and basic attribute setup."""
    test_df, control_df = sample_data_frames
    matcher = Matcher(test=test_df, control=control_df, yvar='is_exposed', exclude=['income'], exhaustive_matching_default=True)

    assert matcher.treatment_col == 'is_exposed'
    assert 'is_exposed' in matcher.data.columns
    assert 'record_id' in matcher.data.columns
    assert matcher.data['record_id'].is_unique
    assert matcher.exhaustive_matching_default is True
    assert 'income' not in matcher.xvars # Due to exclude
    assert 'age' in matcher.xvars
    assert 'feature1' in matcher.xvars
    assert len(matcher.data) == n_test + n_control
    assert len(matcher.test_df) == n_test
    assert len(matcher.control_df) == n_control

def test_fit_scores_and_predict_scores(matcher_instance):
    """Test fitting scores and predicting them."""
    # Mock the actual model fitting to speed up tests and avoid dependency on ML libraries
    mock_model = mock.Mock()
    # Ensure predict_proba returns a 2D array with probabilities for two classes
    mock_model.predict_proba.return_value = np.random.uniform(size=(len(matcher_instance.X), 2))

    with mock.patch('pysmatch.modeling.fit_model', return_value={'model': mock_model, 'accuracy': 0.9}) as mock_fit:
        matcher_instance.fit_scores(balance=False, nmodels=1) # Test with a single model
        mock_fit.assert_called_once() # Check that modeling.fit_model was called

    assert len(matcher_instance.models) == 1
    assert len(matcher_instance.model_accuracy) == 1
    assert matcher_instance.model_accuracy[0] == 0.9

    matcher_instance.predict_scores()
    assert 'scores' in matcher_instance.data.columns
    assert matcher_instance.data['scores'].isnull().sum() == 0
    assert matcher_instance.data['scores'].min() >= 0
    assert matcher_instance.data['scores'].max() <= 1

def test_match_standard_delegation(matcher_instance_fitted):
    """Test that standard matching delegates to matching.perform_match."""
    matcher = matcher_instance_fitted

    # Mock the external matching.perform_match function
    # Assume it returns a DataFrame with specific columns if successful
    mock_matched_df = pd.DataFrame({
        'record_id': [0, 1, 2, 3],
        matcher.treatment_col: [1, 0, 1, 0],
        'scores': [0.6, 0.58, 0.7, 0.72],
        'match_id': [0, 0, 1, 1] # Example output structure
    })

    with mock.patch('pysmatch.matching.perform_match', return_value=mock_matched_df) as mock_perform_match:
        matcher.match(threshold=0.01, nmatches=1, method='nearest', replacement=False, exhaustive_matching=False)
        mock_perform_match.assert_called_once()
        pd.testing.assert_frame_equal(matcher.matched_data, mock_matched_df)

def test_match_exhaustive_matching_basic(matcher_instance_fitted):
    """Test the basic execution of exhaustive matching."""
    matcher = matcher_instance_fitted
    matcher.match(threshold=0.1, nmatches=1, exhaustive_matching=True) # Use a wider threshold for small sample

    assert not matcher.matched_data.empty
    assert 'match_id' in matcher.matched_data.columns
    assert 'record_id' in matcher.matched_data.columns
    assert 'matched_as' in matcher.matched_data.columns
    assert 'pair_score_diff' in matcher.matched_data.columns
    assert matcher.treatment_col in matcher.matched_data.columns

    # Check if pairs are formed correctly (each match_id should have one case and one control)
    if not matcher.matched_data.empty:
        for match_id, group in matcher.matched_data.groupby('match_id'):
            assert len(group) == 2 # Each match_id should have two rows (one case, one control)
            assert set(group['matched_as'].unique()) == {'case', 'control'}
            case_row = group[group['matched_as'] == 'case']
            control_row = group[group['matched_as'] == 'control']
            assert case_row[matcher.treatment_col].iloc[0] == 1
            assert control_row[matcher.treatment_col].iloc[0] == 0


def test_match_exhaustive_nmatches(matcher_instance_fitted):
    """Test that exhaustive matching respects nmatches."""
    matcher = matcher_instance_fitted
    # Ensure there are enough controls to potentially get 2 matches for some test units
    # For this test, we might need to adjust sample data or threshold if it fails due to data sparsity
    matcher.match(threshold=0.2, nmatches=2, exhaustive_matching=True) # Wider threshold

    if not matcher.matched_data.empty:
        # Check if any case was matched to 2 controls (i.e., a case_record_id appears with 2 different match_ids
        # where it's the 'case', and those match_ids have different control_record_ids)
        # This is a bit complex to verify directly without looking at control_usage_counts logic.
        # A simpler check: total rows should be roughly num_test_matched * (1 case + nmatches controls) if all matched
        # Or, count how many unique case_record_ids appear in the 'case' rows of matched_data
        case_ids_in_matched_data = matcher.matched_data[matcher.matched_data['matched_as'] == 'case']['record_id'].unique()

        # For each matched case, it should be associated with `nmatches` controls through distinct `match_id`s
        num_pairs_formed = matcher.matched_data['match_id'].nunique()

        # If all test units found at least one match, num_pairs_formed could be up to testn * nmatches
        # This test is more of an integration test.
        assert num_pairs_formed > 0 # At least some matches
        # A more precise check would be to see if a single case_record_id is part of `nmatches` distinct pairs
        # Example: check one case_record_id
        if len(case_ids_in_matched_data) > 0:
            example_case_id = case_ids_in_matched_data[0]
            num_matches_for_example_case = matcher.matched_data[
                (matcher.matched_data['matched_as'] == 'case') &
                (matcher.matched_data['record_id'] == example_case_id)
                ]['match_id'].nunique()
            # This should be <= nmatches. It might be less if not enough controls are found.
            assert num_matches_for_example_case <= 2


def test_match_exhaustive_threshold(matcher_instance_fitted):
    """Test that exhaustive matching respects the threshold."""
    matcher = matcher_instance_fitted

    # Match with a very strict threshold
    matcher.match(threshold=0.00001, nmatches=1, exhaustive_matching=True)
    # It's likely very few or no matches will be found with such a strict threshold on random scores
    # This test primarily checks that the score_diff in matched_data respects it.
    if not matcher.matched_data.empty:
        assert matcher.matched_data['pair_score_diff'].max() <= 0.00001 + 1e-9 # Add small epsilon for float issues
    else:
        # This is also an acceptable outcome for a very strict threshold
        logging.info("No matches found with very strict threshold, which is expected.")
        pass

def test_record_frequency_exhaustive(matcher_instance_fitted):
    """Test record_frequency after exhaustive matching."""
    matcher = matcher_instance_fitted
    matcher.match(threshold=0.1, nmatches=2, exhaustive_matching=True) # nmatches=2 to encourage reuse

    if not matcher.matched_data.empty:
        freq_df = matcher.record_frequency()
        assert not freq_df.empty
        assert 'record_id' in freq_df.columns
        assert 'n_occurrences_as_control' in freq_df.columns
        # Check if any control was used more than once if nmatches > 1 and data allows
        if matcher.testn > 0 and matcher.controln > 0 and 2 <= matcher.controln / matcher.testn : # crude check if reuse is possible
            # This condition is hard to guarantee without specific data.
            # For now, just check the structure.
            pass
    else:
        freq_df = matcher.record_frequency()
        assert freq_df.empty or list(freq_df.columns) == ['record_id', 'n_occurrences_as_control']


def test_assign_weight_vector(matcher_instance_fitted):
    """Test assigning weights after matching."""
    matcher = matcher_instance_fitted
    matcher.match(threshold=0.1, nmatches=1, exhaustive_matching=True)

    if not matcher.matched_data.empty:
        matcher.assign_weight_vector()
        assert 'weight' in matcher.matched_data.columns
        assert matcher.matched_data['weight'].isnull().sum() == 0
        assert matcher.matched_data['weight'].min() > 0
        assert matcher.matched_data['weight'].max() <= 1.0 # Weight is 1/count, so max is 1 if count=1

        # Verify weight calculation for a specific record if possible
        # e.g., pick a record_id from matched_data, count its occurrences, check weight
        example_record_id = matcher.matched_data['record_id'].iloc[0]
        count = matcher.matched_data[matcher.matched_data['record_id'] == example_record_id].shape[0]
        expected_weight = 1 / count
        actual_weight = matcher.matched_data[matcher.matched_data['record_id'] == example_record_id]['weight'].iloc[0]
        assert np.isclose(actual_weight, expected_weight)

# --- Mocking for visualization and complex external calls ---
# It's good practice to mock plotting functions in unit tests
@mock.patch('pysmatch.visualization.plot_scores')
def test_plot_scores_call(mock_plot, matcher_instance_fitted):
    matcher_instance_fitted.plot_scores()
    mock_plot.assert_called_once()

@mock.patch('pysmatch.visualization.plot_matched_scores')
def test_plot_matched_scores_call(mock_plot, matcher_instance_fitted):
    matcher_instance_fitted.match(exhaustive_matching=True, threshold=0.1) # Ensure matched_data exists
    if not matcher_instance_fitted.matched_data.empty:
        matcher_instance_fitted.plot_matched_scores()
        mock_plot.assert_called_once()
    else:
        logging.info("Skipping plot_matched_scores test as no data was matched.")

@mock.patch('pysmatch.matching.tune_threshold', return_value=(np.array([0.01, 0.02]), np.array([0.8, 0.7])))
@mock.patch('matplotlib.pyplot.show') # Mock plt.show to prevent plots from displaying during tests
def test_tune_threshold_call(mock_plt_show, mock_tune, matcher_instance_fitted):
    matcher_instance_fitted.tune_threshold(method='min')
    mock_tune.assert_called_once()
    # mock_plt_show.assert_called_once() # If plt.show() is always called by tune_threshold's plotting part

# --- Tests for comparison methods (often involve more complex logic or external viz) ---
# These might need more elaborate mocking or to be treated as integration tests

@mock.patch('pysmatch.visualization.compare_continuous', return_value=pd.DataFrame({'var': ['age'], 'smd_before': [0.5], 'smd_after': [0.1]}))
def test_compare_continuous(mock_compare, matcher_instance_fitted):
    matcher = matcher_instance_fitted
    matcher.match(exhaustive_matching=True, threshold=0.1) # Ensure matched_data exists
    result_df = matcher.compare_continuous(return_table=True, plot_result=False) # No plot for unit test
    mock_compare.assert_called_once()
    if not matcher.matched_data.empty:
        assert isinstance(result_df, pd.DataFrame)
        assert 'smd_after' in result_df.columns

@mock.patch('pysmatch.visualization.compare_categorical', return_value=pd.DataFrame({'var': ['feature_cat'], 'p_before': [0.04], 'p_after': [0.5]}))
def test_compare_categorical(mock_compare, matcher_instance_fitted):
    matcher = matcher_instance_fitted
    # Add a categorical variable for testing this
    matcher.data['feature_cat'] = np.random.choice(['A', 'B', 'C'], size=len(matcher.data))
    matcher.xvars.append('feature_cat') # Add to covariates
    matcher.X = matcher.data[matcher.xvars] # Update X
    # Re-fit and predict scores if xvars changed significantly
    # For simplicity, assume it's okay for this test structure

    matcher.match(exhaustive_matching=True, threshold=0.1) # Ensure matched_data exists
    result_df = matcher.compare_categorical(return_table=True, plot_result=False)
    mock_compare.assert_called_once()
    if not matcher.matched_data.empty:
        assert isinstance(result_df, pd.DataFrame)
        assert 'p_after' in result_df.columns

def test_prop_test(matcher_instance_fitted):
    matcher = matcher_instance_fitted
    # Add a categorical variable for testing
    categories = ['cat1', 'cat2', 'cat3']
    matcher.data['categorical_var'] = np.random.choice(categories, size=len(matcher.data))
    matcher.xvars.append('categorical_var') # Add to covariates if it's meant to be one
    # matcher.X = matcher.data[matcher.xvars] # Update X if needed for other parts

    matcher.match(exhaustive_matching=True, threshold=0.1)

    # Test on a categorical variable
    result = matcher.prop_test('categorical_var')
    if result is not None: # Test might return None if conditions for chi2 aren't met
        assert isinstance(result, dict)
        assert result['var'] == 'categorical_var'
        assert 'before' in result
        assert 'after' in result

    # Test on a continuous variable (should return None or log)
    result_cont = matcher.prop_test('age')
    assert result_cont is None

    # Test on excluded variable (should return None or log)
    matcher.exclude.append('feature1') # Temporarily exclude
    result_excluded = matcher.prop_test('feature1')
    assert result_excluded is None
    matcher.exclude.remove('feature1') # Clean up
