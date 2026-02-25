<!-- AUTO-GENERATED: DO NOT EDIT. -->
<!-- Source: ../../../README_CHINESE.md -->

# 端到端流程

## 数据准备

优先选择处理前可观测协变量，避免引入处理后的信息泄漏。

```python
import pandas as pd

fields = [
    "loan_amnt",
    "funded_amnt",
    "funded_amnt_inv",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "loan_status",
]

raw = pd.read_csv("misc/loan.csv", usecols=fields)
test = raw[raw.loan_status == "Default"].copy()
control = raw[raw.loan_status == "Fully Paid"].copy()
```

## 初始化 Matcher

```python
from pysmatch.Matcher import Matcher

matcher = Matcher(
    test=test,
    control=control,
    yvar="is_default",
    exclude=["loan_status"],
)

print("xvars:", matcher.xvars)
print("test/control:", matcher.testn, matcher.controln)
```

## 拟合倾向得分模型

`fit_scores` 支持三类模型：

- `linear`（逻辑回归）
- `knn`
- `tree`（CatBoost，需要安装 `pysmatch[tree]`）

```python
matcher.fit_scores(
    balance=True,
    balance_strategy="over",   # "over" 或 "under"
    nmodels=10,
    model_type="linear",
    max_iter=200,
    n_jobs=2,
)

print("models:", len(matcher.models))
print("avg validation accuracy:", sum(matcher.model_accuracy) / len(matcher.model_accuracy))
```

Optuna 调参路径（训练单个最优模型）：

```python
# matcher.fit_scores(
#     balance=True,
#     model_type="tree",
#     use_optuna=True,
#     n_trials=20,
# )
```

## 预测并查看得分分布

```python
matcher.predict_scores()
matcher.plot_scores()
```

预测后，`matcher.data` 会新增 `scores` 列。

## 调整阈值

```python
import numpy as np

matcher.tune_threshold(
    method="min",
    nmatches=1,
    rng=np.arange(0.0001, 0.0051, 0.0005),
)
```

在保留率与匹配质量之间选择合适阈值。

## 执行匹配

标准匹配：

```python
matcher.match(
    method="min",
    nmatches=1,
    threshold=0.001,
    replacement=False,
    exhaustive_matching=False,
)
matcher.plot_matched_scores()
```

详尽匹配：

```python
matcher.match(
    threshold=0.001,
    nmatches=1,
    exhaustive_matching=True,
)
```

## 检查匹配结果与权重

```python
print(matcher.matched_data.head())
print(matcher.record_frequency().head())
matcher.assign_weight_vector()
print(matcher.matched_data[["record_id", "match_id", "weight"]].head())
```
