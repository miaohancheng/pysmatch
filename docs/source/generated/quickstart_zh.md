<!-- AUTO-GENERATED: DO NOT EDIT. -->
<!-- Source: ../../../README_CHINESE.md -->

# 快速开始

下面是可直接运行的最小示例（使用仓库内置 `misc/loan.csv`）：

```python
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pysmatch.Matcher import Matcher

np.random.seed(42)
data = pd.read_csv("misc/loan.csv")

test = data[data.loan_status == "Default"].copy()
control = data[data.loan_status == "Fully Paid"].copy()

matcher = Matcher(
    test=test,
    control=control,
    yvar="is_default",
    exclude=["loan_status"],
)

matcher.fit_scores(
    balance=True,
    balance_strategy="over",
    nmodels=10,
    model_type="linear",
    n_jobs=2,
)
matcher.predict_scores()
matcher.match(method="min", nmatches=1, threshold=0.001, replacement=False)

print(matcher.matched_data.head())
```

跑通后建议继续阅读完整流程。
