<!-- AUTO-GENERATED: DO NOT EDIT. -->
<!-- Source: ../../../README.md -->

# Matching Strategies

## Standard vs Exhaustive Matching

- **Standard (`exhaustive_matching=False`)**: uses nearest-neighbor style control selection with configurable method/replacement behavior.
- **Exhaustive (`exhaustive_matching=True`)**: prioritizes wider control utilization while still respecting threshold constraints.

## Key Parameters

- `threshold`: max allowed score distance
- `nmatches`: controls per treated unit
- `replacement`: whether a control can be reused
- `method`: `"min"` (closest), `"random"` (random within threshold), or
  `"optimal"` (minimum-cost one-to-one assignment)
- `distance`: `"probability"` or `"logit"` propensity distance; the latter is
  the usual logit-scale caliper
- `exact`: columns that must have identical values within each matched pair

## Practical Guidance

- Start with `nmatches=1`, `replacement=False`, and a moderate threshold.
- If retention is too low, loosen `threshold` gradually.
- If balance is weak after matching, tighten threshold or change model/balance strategy.
- For severe class imbalance, test `balance_strategy="under"` as sensitivity analysis.

To compare matching performance across releases, run the separate benchmark
harness: `python benchmarks/benchmark_matching.py --rows 10000`.

Examples:

```python
# Logit propensity caliper
matcher.match(threshold=0.2, distance="logit")

# Exact matching within site, then nearest propensity score
matcher.match(threshold=0.05, exact=["site"])

# Minimum-cost one-to-one assignment (moderate data sizes)
matcher.match(method="optimal", threshold=0.05)
```

## Large datasets

The default behavior is unchanged. For large frames, process model scoring and
radius-neighbor queries in bounded batches:

```python
matcher.predict_scores(batch_size=50_000)
matcher.match(threshold=0.05, chunk_size=50_000)
```

`chunk_size` limits the treated rows held in each neighbor-query result;
`batch_size` limits rows passed to each model prediction call. Exact matching
and logit calipers can reduce candidate windows further. Exhaustive matching
also accepts `chunk_size`. Global optimal matching still requires a quadratic
cost matrix and should be limited to moderate blocks.
