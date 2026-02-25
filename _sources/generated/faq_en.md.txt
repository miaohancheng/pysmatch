<!-- AUTO-GENERATED: DO NOT EDIT. -->
<!-- Source: ../../../README.md -->

# FAQ

## When should I use `linear` vs `tree` vs `knn`?

- Start with `linear` for strong baseline interpretability.
- Use `tree` for nonlinear relationships and mixed feature types.
- Use `knn` as a local-structure baseline and compare sensitivity.

## Is high model accuracy always better for matching?

Not necessarily. Very high separability may indicate weak overlap, which can reduce matchability. Balance diagnostics matter more than raw classifier accuracy.

## Should I use over- or under-sampling?

- `over`: usually keeps more majority information; good default.
- `under`: faster/smaller training sets; useful for sensitivity checks.

## How do I make runs reproducible?

- set `np.random.seed(...)`
- keep fixed package versions
- record model/matching parameters in experiment logs

## Additional resources

- Sekhon, J. S. (2011), *Multivariate and propensity score matching software with automated balance optimization: The Matching package for R*. Journal of Statistical Software, 42(7), 1-52. [Link](https://www.jstatsoft.org/article/view/v042i07)
- Rosenbaum, P. R., & Rubin, D. B. (1983), *The central role of the propensity score in observational studies for causal effects*. Biometrika, 70(1), 41-55. [Link](https://stat.cmu.edu/~ryantibs/journalclub/rosenbaum_1983.pdf)

## Contributing

Contributions are welcome. Please open an issue or pull request in this repository.

## License

`pysmatch` is released under the MIT License.
