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

## Single Variable Proportion Test

```python
print(matcher.prop_test("grade"))
```
