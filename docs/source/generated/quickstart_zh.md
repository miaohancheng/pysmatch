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

即使只执行干净的 `pip install pysmatch`，上面的示例也可以直接运行。
完整的绘图和诊断流程请参阅 [`Example.py`](https://github.com/miaohancheng/pysmatch/blob/main/Example.py) 或 [`Example.ipynb`](https://github.com/miaohancheng/pysmatch/blob/main/Example.ipynb)。
