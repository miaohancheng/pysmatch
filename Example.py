"""Runnable end-to-end pysmatch example.

Install the complete example dependencies first::

    python -m pip install "pysmatch[all]"
"""

from __future__ import annotations

import warnings

import numpy as np

from pysmatch import Matcher, load_demo

warnings.filterwarnings("ignore")

# 1. Load the compact demo dataset shipped in the wheel.
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

# 2. Fit one reproducible propensity model and score in bounded batches.
matcher.fit_scores(
    balance=True,
    nmodels=2,
    model_type="linear",
    n_jobs=2,
    random_state=42,
)
matcher.predict_scores(batch_size=128)

print("Validation metrics:")
print(matcher.score_report())
print("Calibration:")
print(matcher.calibration_report())
print("Propensity overlap:")
print(matcher.overlap_report())

# 3. Use a logit-scale caliper and chunked nearest-neighbor matching.
matched = matcher.match(
    method="min",
    distance="logit",
    threshold=0.2,
    nmatches=1,
    replacement=False,
    chunk_size=128,
)
print("Matched rows:", len(matched))
print("Balance summary:", matcher.balance_summary())
print(matcher.effective_sample_size())
print(matcher.positivity_report())

# 4. Optional diagnostic plots (requires pysmatch[viz], included by [all]).
matcher.plot_love(threshold=0.1)
matcher.plot_overlap(scope="after")
matcher.plot_calibration()

# 5. Alternative strategies:
# matcher.match(threshold=0.05, exact=["grade"], chunk_size=128)
# matcher.match(method="optimal", threshold=0.05)

# 6. If your input contains an outcome column, estimate a descriptive ATT:
# effect = matcher.estimate_effect("outcome", estimand="ATT", bootstrap=1000)
# print(effect)
