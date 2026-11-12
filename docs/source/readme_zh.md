<!-- AUTO-GENERATED: DO NOT EDIT. -->
<!-- Source: ../../README_CHINESE.md -->

# `pysmatch`

[![PyPI version](https://badge.fury.io/py/pysmatch.svg?icon=si%3Apython&icon_color=%23ffffff)](https://badge.fury.io/py/pysmatch)
[![Downloads](https://static.pepy.tech/badge/pysmatch)](https://pepy.tech/project/pysmatch)
[![CI](https://github.com/miaohancheng/pysmatch/actions/workflows/python-publish.yml/badge.svg)](https://github.com/miaohancheng/pysmatch/actions/workflows/python-publish.yml)
[![Documentation](https://img.shields.io/badge/docs-online-4c8)](https://miaohancheng.com/pysmatch/)
[![GitHub License](https://img.shields.io/github/license/miaohancheng/pysmatch)](LICENSE)
[![codecov](https://codecov.io/github/miaohancheng/pysmatch/graph/badge.svg?token=TUYDEDRV45)](https://codecov.io/github/miaohancheng/pysmatch)

**面向可复现观察性研究的 Python 倾向得分匹配工具。**

`pysmatch` 将倾向得分估计、处理组/对照组匹配和协变量平衡性检查串成一个流程，支持 Logistic Regression、KNN、CatBoost、卡尺匹配、详尽匹配和诊断报告。

`pysmatch` 是 [`pymatch`](https://github.com/benmiroglio/pymatch) 的改进版本，提供更清晰的模块化结构、更灵活的匹配策略以及更完整的评估工具。

### 多语言

[English](https://github.com/miaohancheng/pysmatch/blob/main/README.md) | [中文](https://github.com/miaohancheng/pysmatch/blob/main/README_CHINESE.md)

[在线文档](https://miaohancheng.com/pysmatch/) · [API 文档](https://miaohancheng.com/pysmatch/api.html) · [Issues](https://github.com/miaohancheng/pysmatch/issues) · [引用信息](CITATION.cff)

![内置 demo 数据的匹配前后协变量平衡](docs/source/_static/demo_balance.svg)

### 核心特性

- 多模型打分：Logistic Regression / KNN / CatBoost
- 类别不平衡处理：过采样与欠采样（`balance_strategy`）
- 标准匹配与详尽匹配两种路径
- 连续/分类协变量平衡性诊断
- 可选 Optuna 自动调参

## 安装

从 PyPI 安装：

```bash
pip install pysmatch
```

安装可选依赖：

```bash
pip install "pysmatch[tree]"   # 启用 CatBoost
pip install "pysmatch[tune]"   # 启用 Optuna
pip install "pysmatch[viz]"    # 绘图和平衡性可视化
pip install "pysmatch[all]"    # 启用全部可选依赖
pip install "pysmatch[dev]"    # 测试、检查和打包工具
```

从源码安装：

```bash
git clone https://github.com/miaohancheng/pysmatch.git
cd pysmatch
pip install -e ".[all]"
```

## 快速开始

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

## 端到端流程

### 数据准备

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

### 初始化 Matcher

```python
from pysmatch import Matcher

matcher = Matcher(
    test=test,
    control=control,
    yvar="is_default",
    exclude=["loan_status"],
)

print("xvars:", matcher.xvars)
print("test/control:", matcher.testn, matcher.controln)
```

### 拟合倾向得分模型

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

### 预测并查看得分分布

```python
matcher.predict_scores()
matcher.plot_scores()
```

预测后，`matcher.data` 会新增 `scores` 列。

### 调整阈值

```python
import numpy as np

matcher.tune_threshold(
    method="min",
    nmatches=1,
    rng=np.arange(0.0001, 0.0051, 0.0005),
)
```

在保留率与匹配质量之间选择合适阈值。

### 执行匹配

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

### 检查匹配结果与权重

```python
print(matcher.matched_data.head())
print(matcher.record_frequency().head())
matcher.assign_weight_vector()
print(matcher.matched_data[["record_id", "match_id", "weight"]].head())
```

## 匹配策略

### 标准匹配 vs 详尽匹配

- **标准匹配（`exhaustive_matching=False`）**：按方法和阈值选择控制样本，行为更接近经典最近邻。
- **详尽匹配（`exhaustive_matching=True`）**：优先提高控制组利用率，在阈值内尽可能形成高质量配对。

### 关键参数

- `threshold`：可接受的最大得分差
- `nmatches`：每个处理样本匹配的控制样本数量
- `replacement`：控制样本是否可重复使用
- `method`：`"min"`（最近）、`"random"`（阈值内随机）或
  `"optimal"`（最小成本的一对一分配）
- `distance`：`"probability"` 或 `"logit"` 倾向得分距离，后者是常见的 logit caliper
- `exact`：要求匹配对中取值完全相同的列

### 实践建议

- 先用 `nmatches=1`、`replacement=False` 和中等阈值起步。
- 若保留率太低，逐步放宽 `threshold`。
- 若匹配后仍不平衡，适当收紧阈值或更换模型。
- 类别极度不平衡时，可将 `balance_strategy="under"` 作为稳健性对照。

如需比较不同版本的匹配性能，可运行独立 benchmark：
`python benchmarks/benchmark_matching.py --rows 10000`。

示例：

```python
# logit 倾向得分 caliper
matcher.match(threshold=0.2, distance="logit")

# 在 site 内做 exact matching，再按倾向得分最近匹配
matcher.match(threshold=0.05, exact=["site"])

# 中等规模数据上的最小成本一对一匹配
matcher.match(method="optimal", threshold=0.05)
```

### 大规模数据集

默认行为保持不变。对于大表，可以限制模型预测和最近邻查询的批次大小：

```python
matcher.predict_scores(batch_size=50_000)
matcher.match(threshold=0.05, chunk_size=50_000)
```

`batch_size` 限制每次模型预测的行数，`chunk_size` 限制每次最近邻查询保留的
treated 行数。Exact matching 和 logit caliper 可以进一步缩小候选窗口。
Exhaustive matching 也支持 `chunk_size`；全局 optimal matching 仍需要二次规模
成本矩阵，应限制在中等规模分块内。

## 评估

匹配完成后，应先做平衡性评估，再做后续因果分析。

表格形式的 `balance_report()` 只需要核心安装；`compare_continuous()`、
`plot_scores()` 等绘图函数需要额外安装 `pip install "pysmatch[viz]"`。

### 分类变量平衡

```python
cat_table = matcher.compare_categorical(return_table=True, plot_result=True)
print(cat_table)
```

关注点：

- 匹配前后 p 值变化
- 比例差异是否明显下降

### 连续变量平衡

```python
cont_table = matcher.compare_continuous(return_table=True, plot_result=True)
print(cont_table)
```

关注点：

- KS 与 grouped permutation 检验结果
- 标准化均值差/中位数差在匹配后是否下降

也可以使用统一的平衡性报告，把标准化差异作为主要指标：

```python
balance = matcher.balance_report()
print(balance[["variable", "abs_smd_before", "abs_smd_after"]])
```

绝对 SMD 应结合保留率和研究设计一起解释，不能只看显著性检验。

解释效应前可以先检查倾向得分的共同支持区间：

```python
print(matcher.overlap_report())
```

其他结构化诊断也可以直接调用，无需解析图形：

```python
print(matcher.positivity_report())
print(matcher.effective_sample_size())
print(matcher.score_report())       # accuracy、AUC 和 Brier score
print(matcher.calibration_report())
matcher.plot_love(threshold=0.1)
matcher.plot_overlap(scope="after")
matcher.plot_calibration()
```

### 估计匹配后的处理效应

确认重叠性和平衡性后，可以对匹配样本估计处理效应，并用 bootstrap
计算可复现的区间：

```python
effect = matcher.estimate_effect(
    outcome="outcome",
    estimand="ATT",       # "ATT"、"ATC" 或 "ATE"
    bootstrap=1000,
    random_state=42,
)
print(effect["estimate"], effect["ci_lower"], effect["ci_upper"])
```

该结果仍依赖研究设计的识别假设，不能消除未观测混杂。报告时应同时说明
estimand、保留率、平衡性和样本定义。

### 单变量比例检验

```python
print(matcher.prop_test("grade"))
```

## 故障排查

### `ValueError: numpy.dtype size changed`

通常是 NumPy/Pandas 二进制兼容性问题。

```bash
pip install --upgrade --force-reinstall "numpy>=1.26.4" "pandas>=2.1.4"
```

重装后请重启 Python 内核或会话。

### `Scores column not found`

调用顺序应为：先拟合打分模型，再预测分数，再匹配。

```python
matcher.fit_scores(...)
matcher.predict_scores()
matcher.match(...)
```

### 读取数据报 `FileNotFoundError`

内置 demo 数据无需仓库路径即可读取：

```python
from pysmatch import load_demo
data = load_demo()
```

### 匹配结果为空

常见原因是阈值过严或组间重叠不足。

- 适度增大 `threshold`
- 更换 `model_type`
- 先用 `plot_scores()` 检查组间得分重叠

### Notebook 内核不可用

若 notebook 的 kernelspec 不存在，请切换到可用内核（如 `python3`）后重跑。

## 常见问题

### 何时用 `linear`、`tree`、`knn`？

- `linear`：默认首选，解释性强，稳健。
- `tree`：适合非线性关系和复杂特征结构。
- `knn`：适合做局部结构对照和敏感性比较。

### 模型准确率越高越好吗？

不一定。准确率过高有时意味着组间重叠不足，反而会降低可匹配性。平衡性指标比单纯准确率更关键。

### 过采样还是欠采样怎么选？

- `over`：通常保留更多多数类信息，建议先用。
- `under`：训练更快，适合作为稳健性对照。

### 如何保证可复现？

- 给 `Matcher` 和匹配函数传入固定的 `random_state`
- 固定依赖版本
- 记录完整的匹配参数与模型参数

### 延伸阅读

- Sekhon, J. S. (2011), *Multivariate and propensity score matching software with automated balance optimization: The Matching package for R*. Journal of Statistical Software, 42(7), 1-52. [Link](https://www.jstatsoft.org/article/view/v042i07)
- Rosenbaum, P. R., & Rubin, D. B. (1983), *The central role of the propensity score in observational studies for causal effects*. Biometrika, 70(1), 41-55. [Link](https://stat.cmu.edu/~ryantibs/journalclub/rosenbaum_1983.pdf)

### 贡献

欢迎提交 issue 和 pull request。

### 引用

请参阅仓库中的 `CITATION.cff` 获取引用信息；发表结果时也请引用下方的方法学文献。

### 许可证

`pysmatch` 使用 MIT License。
