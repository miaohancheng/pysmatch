# Contributing to pysmatch

Start with a focused issue or discussion for larger API and statistical changes. Small bug fixes and documentation improvements can go directly through a pull request.

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[all,dev]"
.venv/bin/python scripts/sync_readme_to_docs.py --check
MPLBACKEND=Agg .venv/bin/python -m pytest
```

Please include a minimal reproducible example for bug fixes and tests for changed behavior. Changes to matching or balance metrics should document the estimand, assumptions, and any compatibility impact. Keep public API additions documented in both the English and Chinese README where applicable.

The project uses the MIT license. By contributing, you agree that your contributions are licensed under the same terms.
