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
