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

## 单变量比例检验

```python
print(matcher.prop_test("grade"))
```
