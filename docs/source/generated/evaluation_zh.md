<!-- AUTO-GENERATED: DO NOT EDIT. -->
<!-- Source: ../../../README_CHINESE.md -->

# 评估

匹配完成后，应先做平衡性评估，再做后续因果分析。

表格形式的 `balance_report()` 只需要核心安装；`compare_continuous()`、
`plot_scores()` 等绘图函数需要额外安装 `pip install "pysmatch[viz]"`。

## 分类变量平衡

```python
cat_table = matcher.compare_categorical(return_table=True, plot_result=True)
print(cat_table)
```

关注点：

- 匹配前后 p 值变化
- 比例差异是否明显下降

## 连续变量平衡

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

## 估计匹配后的处理效应

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

## 单变量比例检验

```python
print(matcher.prop_test("grade"))
```
