# Example.py
import pandas as pd
import numpy as np
from pysmatch.Matcher import Matcher # Assuming Matcher.py is in pysmatch directory

# --- 1. Generate Sample Data ---
# Create more distinct data for test and control to better see matching effects
np.random.seed(42)
n_test = 100
n_control = 300

# Test Data (e.g., treated group)
test_data = pd.DataFrame({
    'age': np.random.normal(loc=50, scale=5, size=n_test),
    'income': np.random.normal(loc=60000, scale=10000, size=n_test),
    'education': np.random.choice([12, 16, 18], size=n_test, p=[0.3, 0.5, 0.2]),
    'health_score': np.random.uniform(60, 90, size=n_test) # Higher health for test
})
# test_data['treatment'] = 1 # Treatment column will be added by Matcher

# Control Data
control_data = pd.DataFrame({
    'age': np.random.normal(loc=45, scale=7, size=n_control),
    'income': np.random.normal(loc=50000, scale=12000, size=n_control),
    'education': np.random.choice([10, 12, 16], size=n_control, p=[0.4, 0.4, 0.2]),
    'health_score': np.random.uniform(40, 75, size=n_control) # Lower health for control
})
# control_data['treatment'] = 0 # Treatment column will be added by Matcher

print("--- Sample Test Data ---")
print(test_data.head())
print("\n--- Sample Control Data ---")
print(control_data.head())

# --- 2. Initialize Matcher ---
# Define the treatment variable name (this column will be created by Matcher)
treatment_var = 'is_treated'
# Define covariates to be used for matching
# 'record_id', 'scores', 'match_id', 'weight' etc. are automatically handled by exclude in Matcher
# Ensure no NA values in these covariates in your real data, or handle them before passing.
# For this example, data generation avoids NAs in these key columns.

# Initialize Matcher with default exhaustive_matching_default=False
matcher_instance = Matcher(test=test_data,
                           control=control_data,
                           yvar=treatment_var,
                           exclude=['some_other_column_to_exclude']) # Example of an excluded column

print(f"\n--- Matcher Initialized ---")
print(f"Treatment Column: {matcher_instance.treatment_col}")
print(f"Covariates (xvars): {matcher_instance.xvars}")
print(f"Initial data shape: {matcher_instance.data.shape}")
print(f"Test group size: {matcher_instance.testn}, Control group size: {matcher_instance.controln}")

# --- 3. Fit Propensity Score Model(s) ---
# Using logistic regression by default ('linear')
# You can try 'knn' (K-nearest neighbors),  'tree' (Catboost) for model_type
matcher_instance.fit_scores(balance=True, nmodels=5, model_type='linear')
print("\n--- Propensity Score Models Fitted ---")
if matcher_instance.models:
    print(f"Number of models fitted: {len(matcher_instance.models)}")
    print(f"Average model accuracy: {np.mean(matcher_instance.model_accuracy):.4f}")
else:
    print("No models were fitted.")

# --- 4. Predict Propensity Scores ---
matcher_instance.predict_scores()
print("\n--- Propensity Scores Predicted ---")
if 'scores' in matcher_instance.data.columns:
    print("Scores column added to internal data:")
    print(matcher_instance.data[['record_id', treatment_var, 'scores']].head())
else:
    print("Scores column not found.")

# --- 5. Plot Score Distributions (Before Matching) ---
print("\n--- Plotting Score Distributions (Before Matching) ---")
matcher_instance.plot_scores() # This will show a plot

# --- 6. Tune Threshold (Optional) ---
# This helps decide a good threshold value by seeing retention rates
print("\n--- Tuning Threshold (Example) ---")

try:
    matcher_instance.tune_threshold(method='min', nmatches=1, rng=np.arange(0.0001, 0.0051, 0.0005))
except Exception as e:
    print(f"Could not run tune_threshold, possibly due to external 'matching' module: {e}")


# --- 7. Perform Matching ---

# Example 7a: Standard Matching (using pysmatch.matching.perform_match via Matcher)
print("\n--- Performing Standard Matching (exhaustive_matching=False) ---")
matcher_instance.match(threshold=0.002, nmatches=1, method='min', replacement=False, exhaustive_matching=False)
if not matcher_instance.matched_data.empty:
    print(f"Standard matching complete. Matched data shape: {matcher_instance.matched_data.shape}")
    print("Sample of standard matched data:")
    print(matcher_instance.matched_data[['record_id', treatment_var, 'scores', 'match_id']].head()) # 'match_id' might be named differently by external func
else:
    print("No matches found with standard matching.")

# Example 7b: Exhaustive Matching
print("\n--- Performing Exhaustive Matching ---")
# For exhaustive matching, 'method' and 'replacement' are handled internally by the new logic.
# Set exhaustive_matching_default=True in __init__ or exhaustive_matching=True in match()
matcher_instance_exhaustive = Matcher(test=test_data,
                                      control=control_data,
                                      yvar=treatment_var,
                                      exhaustive_matching_default=True)
matcher_instance_exhaustive.fit_scores(balance=True, nmodels=5)
matcher_instance_exhaustive.predict_scores()
matcher_instance_exhaustive.match(threshold=0.002, nmatches=2) # Requesting 2 matches per test unit

if not matcher_instance_exhaustive.matched_data.empty:
    print(f"Exhaustive matching complete. Matched data shape: {matcher_instance_exhaustive.matched_data.shape}")
    print("Sample of exhaustive matched data (long format):")
    # Columns from exhaustive matching: record_id, match_id, matched_as, pair_score_diff, and original data
    print(matcher_instance_exhaustive.matched_data[['record_id', treatment_var, 'scores', 'match_id', 'matched_as', 'pair_score_diff']].head())

    # --- 8. Plot Matched Score Distributions ---
    print("\n--- Plotting Matched Score Distributions (Exhaustive) ---")
    matcher_instance_exhaustive.plot_matched_scores() # This will show a plot

    # --- 9. Record Frequency and Weighting (Example with Exhaustive Matched Data) ---
    print("\n--- Record Frequency (Exhaustive Matched Controls) ---")
    # This shows how many times each original control record was used
    control_usage_freq = matcher_instance_exhaustive.record_frequency()
    print(control_usage_freq.head())

    print("\n--- Assigning Weights (Exhaustive Matched Data) ---")
    matcher_instance_exhaustive.assign_weight_vector()
    if 'weight' in matcher_instance_exhaustive.matched_data.columns:
        print("Weights assigned. Sample:")
        print(matcher_instance_exhaustive.matched_data[['record_id', 'match_id', 'weight', 'matched_as']].head())
    else:
        print("Weight column not found after assignment.")

    # --- 10. Covariate Balance Assessment (Example with Exhaustive Matched Data) ---
    # Ensure visualization.compare_continuous and compare_categorical are implemented
    print("\n--- Comparing Continuous Variables (Exhaustive Matched Data) ---")
    try:
        continuous_comparison = matcher_instance_exhaustive.compare_continuous(return_table=True, plot_result=True)
        if continuous_comparison is not None:
            print("Continuous variables comparison table:")
            print(continuous_comparison)
    except Exception as e:
        print(f"Could not compare continuous variables: {e}")


    print("\n--- Comparing Categorical Variables (Exhaustive Matched Data) ---")

    education_prop_test = matcher_instance_exhaustive.prop_test('education')
    if education_prop_test:
        print("Prop_test results for 'education':")
        print(education_prop_test)

    try:
        categorical_comparison = matcher_instance_exhaustive.compare_categorical(return_table=True, plot_result=True)
        if categorical_comparison is not None:
            print("Categorical variables comparison table:")
            print(categorical_comparison) # This table would summarize tests like the one above
    except Exception as e:
        print(f"Could not compare categorical variables: {e}")

else:
    print("No matches found with exhaustive matching, skipping further analysis steps.")

print("\n--- Example Script Finished ---")
