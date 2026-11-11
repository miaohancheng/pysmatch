.. _usage:

Usage Guide
===========

The shortest complete workflow is available in the generated
`Quickstart <generated/quickstart_en.html>`_ page. The same example works from
a clean PyPI installation because the demo dataset is packaged with pysmatch.

.. code-block:: python

   from pysmatch import Matcher, load_demo

   data = load_demo()
   test = data[data.loan_status == "Default"].copy()
   control = data[data.loan_status == "Fully Paid"].copy()

   matcher = Matcher(
       test=test,
       control=control,
       yvar="is_default",
       exclude=["loan_status"],
       random_state=42,
   )
   matcher.fit_scores(model_type="linear", balance=True, nmodels=10, n_jobs=2)
   matcher.predict_scores()
   matched = matcher.match(threshold=0.001, nmatches=1, replacement=False)
   print(matched.head())
   print(matcher.balance_report())

Use ``model_type="tree"`` after installing ``pysmatch[tree]`` for CatBoost,
or ``use_optuna=True`` after installing ``pysmatch[tune]`` for tuning. Always
inspect balance and retention before using matched data for causal analysis.
