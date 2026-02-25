<!-- AUTO-GENERATED: DO NOT EDIT. -->
<!-- Source: ../../../README.md -->

# Troubleshooting

## `ValueError: numpy.dtype size changed`

This is usually a NumPy/Pandas binary compatibility issue.

```bash
pip install --upgrade --force-reinstall "numpy>=1.26.4" "pandas>=2.1.4"
```

Restart your Python kernel/session after reinstalling.

## `Scores column not found`

Run `predict_scores()` before `match()`.

```python
matcher.fit_scores(...)
matcher.predict_scores()
matcher.match(...)
```

## `FileNotFoundError` for dataset path

Use a repo-relative path:

```python
pd.read_csv("misc/loan.csv")
```

## No matches found

Usually threshold is too strict or groups are weakly overlapping.

- increase `threshold`
- try a different `model_type`
- inspect score distributions with `plot_scores()`

## Jupyter kernel issues in notebooks

If your notebook kernel name is unavailable, switch to an existing kernel (`python3`) and rerun cells.
