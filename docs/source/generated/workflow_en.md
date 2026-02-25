<!-- AUTO-GENERATED: DO NOT EDIT. -->
<!-- Source: ../../../README.md -->

# End-to-End Workflow

## Data Preparation

Use domain-relevant covariates and avoid leaking post-treatment variables into matching features.

```python
import pandas as pd

fields = [
    "loan_amnt",
    "funded_amnt",
    "funded_amnt_inv",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "loan_status",
]

raw = pd.read_csv("misc/loan.csv", usecols=fields)
test = raw[raw.loan_status == "Default"].copy()
control = raw[raw.loan_status == "Fully Paid"].copy()
```

## Initialize Matcher

```python
from pysmatch.Matcher import Matcher

matcher = Matcher(
    test=test,
    control=control,
    yvar="is_default",
    exclude=["loan_status"],
)

print("xvars:", matcher.xvars)
print("test/control:", matcher.testn, matcher.controln)
```

## Fit Propensity Score Models

`fit_scores` supports three model types:

- `linear` (logistic regression)
- `knn`
- `tree` (CatBoost, requires `pysmatch[tree]`)

```python
matcher.fit_scores(
    balance=True,
    balance_strategy="over",   # "over" or "under"
    nmodels=10,
    model_type="linear",
    max_iter=200,
    n_jobs=2,
)

print("models:", len(matcher.models))
print("avg validation accuracy:", sum(matcher.model_accuracy) / len(matcher.model_accuracy))
```

Optuna path (single tuned model):

```python
# matcher.fit_scores(
#     balance=True,
#     model_type="tree",
#     use_optuna=True,
#     n_trials=20,
# )
```

## Predict and Plot Scores

```python
matcher.predict_scores()
matcher.plot_scores()
```

`matcher.data` now contains a `scores` column.

## Tune Threshold

```python
import numpy as np

matcher.tune_threshold(
    method="min",
    nmatches=1,
    rng=np.arange(0.0001, 0.0051, 0.0005),
)
```

Choose a threshold that balances quality and retained sample size.

## Run Matching

Standard matching:

```python
matcher.match(
    method="min",
    nmatches=1,
    threshold=0.001,
    replacement=False,
    exhaustive_matching=False,
)
matcher.plot_matched_scores()
```

Exhaustive matching:

```python
matcher.match(
    threshold=0.001,
    nmatches=1,
    exhaustive_matching=True,
)
```

## Review Matched Data and Weights

```python
print(matcher.matched_data.head())
print(matcher.record_frequency().head())
matcher.assign_weight_vector()
print(matcher.matched_data[["record_id", "match_id", "weight"]].head())
```
