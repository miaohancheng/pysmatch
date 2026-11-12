<!-- AUTO-GENERATED: DO NOT EDIT. -->
<!-- Source: ../../../README.md -->

# Quickstart

This minimal example runs the full core path with the demo dataset shipped in the package.

```python
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from pysmatch import Matcher, load_demo

np.random.seed(42)
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

matcher.fit_scores(
    balance=True,
    balance_strategy="over",
    nmodels=10,
    model_type="linear",
    n_jobs=2,
)
matcher.predict_scores(batch_size=128)
matcher.match(
    method="min",
    distance="logit",
    threshold=0.2,
    nmatches=1,
    replacement=False,
    chunk_size=128,
)

print(matcher.matched_data.head())
print(matcher.balance_report())
print(matcher.overlap_report())
print(matcher.effective_sample_size())
```

The demo data is shipped with the package, so this example also works after a clean `pip install pysmatch`.
For a complete plotting and diagnostics walkthrough, see [`Example.py`](https://github.com/miaohancheng/pysmatch/blob/main/Example.py) or [`Example.ipynb`](https://github.com/miaohancheng/pysmatch/blob/main/Example.ipynb).
