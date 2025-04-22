.. _usage:

Usage Guide
===========

Provide examples and guides on how to use the core functionalities of pysmatch.

Basic Example
-------------

.. code-block:: python

   import pandas as pd
   from pysmatch import Matcher

   # Load your data
   # data = pd.read_csv(...)
   # treated_col = 'treatment_variable_name'
   # outcome_col = 'outcome_variable_name'
   # covariates = ['cov1', 'cov2', ...]

   # Assuming 'data', 'treated_col', 'outcome_col', 'covariates' are defined
   # m = Matcher(data, treated_col, outcome_col, covariates)
   # m.fit_scores(balance=True, n_folds=5) # Example
   # m.match(method="nearest", n_matches=1, caliper=0.1)
   # m.predict_scores() # Or fit_scores if not done
   # results = m.compare_continuous(method="ttest")
   # print(results)


Refer to the :doc:`example_notebook` for a complete walkthrough.