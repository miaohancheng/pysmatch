.. _usage:

Usage Guide
===========

Provide examples and guides on how to use the core functionalities of pysmatch.

Basic Example
-------------

.. code-block:: python

        import pandas as pd
        import numpy as np
        from pysmatch.Matcher import Matcher # Assuming Matcher.py is in pysmatch directory
        import warnings

        # --- 0. Suppress Warnings (Optional) ---
        warnings.filterwarnings('ignore')

        # --- 1. Load and Prepare Data ---
        # Load loan.csv data as per user's request
        path = "misc/loan.csv" # Ensure this path is correct or the file is in the same directory
        data = pd.read_csv(path)

        test_data_full = data[data.loan_status == "Default"].copy() # Use .copy() to avoid SettingWithCopyWarning
        control_data_full = data[data.loan_status == "Fully Paid"].copy()

        # --- 2. Initialize Matcher ---
        # Define the treatment variable name (Matcher will create this column)
        treatment_var = 'is_default' # Any descriptive name; Matcher will use this as the column name

        # Define covariates for matching.
        # Select suitable columns from loan.csv as covariates.
        # E.g., "loan_amnt", "funded_amnt", "int_rate", "installment", "grade"
        # Columns like 'record_id', 'scores', 'match_id', 'weight' are handled by Matcher's exclude logic.
        # Ensure no NA values in these covariates in your real data, or handle them before passing.
        # Matcher also has internal NA handling for xvars.

        # Exclude the original 'loan_status' string column as we use 'is_default' as the treatment indicator.
        # You can add other ID-like columns or columns unsuitable as covariates to exclude_cols.
        exclude_cols = ['loan_status']

        matcher_instance = Matcher(test=test_data_full,
                                   control=control_data_full,
                                   yvar=treatment_var, # Matcher will create 'is_default' column internally
                                   exclude=exclude_cols)

        print(f"\n--- Matcher Initialized ---")
        print(f"Treatment Column: {matcher_instance.treatment_col}")
        print(f"Covariates (xvars) selected by Matcher: {matcher_instance.xvars}")
        print(f"Initial data shape in Matcher: {matcher_instance.data.shape}")
        print(f"Test group size in Matcher: {matcher_instance.testn}, Control group size in Matcher: {matcher_instance.controln}")

        # --- 3. Fit Propensity Score Model(s) ---
        # Available model_type options: 'linear' (Logistic Regression), 'knn' (K-Nearest Neighbors),'tree' (CatBoost).
        # nmodels is recommended to be adjusted based on the sample size of the minority class.
        # Using n_jobs > 1 for parallel processing if your machine has multiple cores.
        if matcher_instance.testn > 0 and matcher_instance.controln > 0:
            matcher_instance.fit_scores(balance=True, nmodels=10, model_type='tree', n_jobs=2)
            # Example with another model type:
            # matcher_instance.fit_scores(balance=True, nmodels=5, model_type='tree', n_jobs=2) # Using CatBoost
            # Example with Optuna for hyperparameter tuning:
            # matcher_instance.fit_scores(balance=True, model_type='tree', use_optuna=True, n_trials=20, n_jobs=2)

            print("\n--- Propensity Score Models Fitted ---")
            print(f"Number of models fitted: {len(matcher_instance.models)}")
            print(f"Average model accuracy: {np.mean(matcher_instance.model_accuracy):.4f}")

        # --- 4. Predict Propensity Scores ---
        if matcher_instance.models: # Only predict if models were fitted
            matcher_instance.predict_scores()
            print("\n--- Propensity Scores Predicted ---")
            print("Scores column added to internal data:")
            print(matcher_instance.data[['record_id', matcher_instance.treatment_col, 'scores']].head())


        # --- 5. Plot Score Distributions (Before Matching) ---
        print("\n--- Plotting Score Distributions (Before Matching) ---")
        matcher_instance.plot_scores() # This will show a plot


        # --- 6. Tune Threshold (Optional) ---
        # This helps decide a good threshold value by seeing retention rates.
        matcher_instance.tune_threshold(method='min', nmatches=1, rng=np.arange(0.0001, 0.0051, 0.0005))
        # --- 7. Perform Matching ---
        # The match method offers flexibility:
        # - Standard matching: `exhaustive_matching=False` (default).
        #   - `method`: 'min', 'random', 'nn' (if implemented in pysmatch.matching).
        #   - `replacement`: True or False.
        # - Exhaustive matching: `exhaustive_matching=True`.
        #   - `method` and `replacement` parameters are ignored for exhaustive matching as its logic is internal.

        # Example 7a: Standard Matching with replacement
        print("\n--- Performing Standard Matching (exhaustive_matching=False, replacement=True) ---")
        matcher_instance.match(threshold=0.001, nmatches=1, method='min', replacement=True, exhaustive_matching=False)
        print(f"Standard matching (with replacement) complete. Matched data shape: {matcher_instance.matched_data.shape}")
        print("Sample of standard matched data:")
        display_cols_standard = ['record_id', matcher_instance.treatment_col, 'scores']
        if 'match_id' in matcher_instance.matched_data.columns: # 'match_id' might be named differently by external func
            display_cols_standard.append('match_id')
        if 'weight' in matcher_instance.matched_data.columns: # Weights are often added after matching with replacement
            display_cols_standard.append('weight')
        print(matcher_instance.matched_data[display_cols_standard].head())

        # Example 7b: Exhaustive Matching
        print("\n--- Performing Exhaustive Matching (exhaustive_matching=True) ---")
        # For exhaustive matching, 'method' and 'replacement' are handled internally by the new logic.
        # You can set exhaustive_matching_default=True in __init__ or exhaustive_matching=True in match() call.

        # Re-matching on the same instance (matched_data will be overwritten)
        # Or, create a new Matcher instance if you want to keep the previous standard match results.
        matcher_instance.match(threshold=0.001, nmatches=1, exhaustive_matching=True) # Requesting 1 match per test unit
        print(f"Exhaustive matching complete. Matched data shape: {matcher_instance.matched_data.shape}")
        print("Sample of exhaustive matched data (long format):")
        display_cols_exhaustive = ['record_id', matcher_instance.treatment_col, 'scores', 'match_id', 'matched_as', 'pair_score_diff']
        actual_display_cols_exhaustive = [col for col in display_cols_exhaustive if col in matcher_instance.matched_data.columns]
        print(matcher_instance.matched_data[actual_display_cols_exhaustive].head())

        # --- 8. Plot Matched Score Distributions (after Exhaustive Matching) ---
        print("\n--- Plotting Matched Score Distributions (Exhaustive) ---")
        matcher_instance.plot_matched_scores() # This will show a plot

        # --- 9. Record Frequency and Weighting (Example with Exhaustive Matched Data) ---
        print("\n--- Record Frequency (Controls in Exhaustive Matched Data) ---")
        # This shows how many times each original control record was used
        control_usage_freq = matcher_instance.record_frequency()
        print(control_usage_freq.head())

        print("\n--- Assigning Weights (Exhaustive Matched Data) ---")
        matcher_instance.assign_weight_vector()
        print("Weights assigned. Sample:")
        weight_display_cols = ['record_id', 'match_id', 'weight']
        if 'matched_as' in matcher_instance.matched_data.columns:
            weight_display_cols.append('matched_as')
        actual_weight_display_cols = [col for col in weight_display_cols if col in matcher_instance.matched_data.columns]
        print(matcher_instance.matched_data[actual_weight_display_cols].head())


        # --- 10. Covariate Balance Assessment (Example with Exhaustive Matched Data) ---
        # Ensure visualization.compare_continuous and compare_categorical are implemented
        # and that the Matcher instance has the necessary attributes (self.test, self.control, self.yvar)
        print("\n--- Comparing Continuous Variables (Exhaustive Matched Data) ---")
        continuous_comparison = matcher_instance.compare_continuous(return_table=True, plot_result=True)
        print("Continuous variables comparison table:")
        print(continuous_comparison)



        print("\n--- Comparing Categorical Variables (Exhaustive Matched Data) ---")
        # Example: 'grade' is a categorical variable in loan.csv data
        # prop_test is for individual categorical variables.
        # compare_categorical would ideally iterate through them.
        grade_prop_test = matcher_instance.prop_test('grade')
        print("Prop_test results for 'grade':")
        print(grade_prop_test)


        categorical_comparison = matcher_instance.compare_categorical(return_table=True, plot_result=True)
        print("Categorical variables comparison table:")
        print(categorical_comparison)

        print("\n--- Example Script Finished ---")


