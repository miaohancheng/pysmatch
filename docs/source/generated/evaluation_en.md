<!-- AUTO-GENERATED: DO NOT EDIT. -->
<!-- Source: ../../../README.md -->

# Evaluation

After matching, evaluate covariate balance before causal analysis.

The table-based `balance_report()` works with the core install. Plotting
helpers such as `compare_continuous()` and `plot_scores()` require
`pip install "pysmatch[viz]"`.

## Categorical Covariates

```python
cat_table = matcher.compare_categorical(return_table=True, plot_result=True)
print(cat_table)
```

Interpretation:

- check before/after p-value shifts
- look for reduced proportional differences after matching

## Continuous Covariates

```python
cont_table = matcher.compare_continuous(return_table=True, plot_result=True)
print(cont_table)
```

Interpretation:

- compare KS statistics and grouped permutation test p-values
- monitor standardized mean/median differences pre vs post matching

For a compact report that uses standardized mean differences (SMD) and
category-proportion differences as the primary balance measures:

```python
balance = matcher.balance_report()
print(balance[["variable", "abs_smd_before", "abs_smd_after"]])
```

As a rule of thumb, inspect absolute SMD values together with retention and
the research design; statistical significance alone is not a balance target.

Check common support before interpreting an effect:

```python
print(matcher.overlap_report())
```

The other structured diagnostics are available without parsing plots:

```python
print(matcher.positivity_report())
print(matcher.effective_sample_size())
print(matcher.score_report())       # accuracy, AUC, and Brier score
print(matcher.calibration_report())
matcher.plot_love(threshold=0.1)
matcher.plot_overlap(scope="after")
matcher.plot_calibration()
```

## Estimate a Matched Effect

After checking overlap and balance, estimate a descriptive matched-sample
contrast with a reproducible bootstrap interval:

```python
effect = matcher.estimate_effect(
    outcome="outcome",
    estimand="ATT",       # "ATT", "ATC", or "ATE"
    bootstrap=1000,
    random_state=42,
)
print(effect["estimate"], effect["ci_lower"], effect["ci_upper"])
```

This estimate still relies on the study's identification assumptions and does
not remove unmeasured confounding. Report the estimand, retention, balance, and
sample definition together.

## Single Variable Proportion Test

```python
print(matcher.prop_test("grade"))
```
