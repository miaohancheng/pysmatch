<!-- AUTO-GENERATED: DO NOT EDIT. -->
<!-- Source: ../../../README_CHINESE.md -->

# 快速开始

下面是可直接运行的最小示例（demo 数据随包发布）：

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
matcher.predict_scores()
matcher.match(method="min", nmatches=1, threshold=0.001, replacement=False)

print(matcher.matched_data.head())
print(matcher.balance_report())
```

即使只执行干净的 `pip install pysmatch`，上面的示例也可以直接运行。
