<!-- AUTO-GENERATED: DO NOT EDIT. -->
<!-- Source: ../../../README_CHINESE.md -->

# 匹配策略

## 标准匹配 vs 详尽匹配

- **标准匹配（`exhaustive_matching=False`）**：按方法和阈值选择控制样本，行为更接近经典最近邻。
- **详尽匹配（`exhaustive_matching=True`）**：优先提高控制组利用率，在阈值内尽可能形成高质量配对。

## 关键参数

- `threshold`：可接受的最大得分差
- `nmatches`：每个处理样本匹配的控制样本数量
- `replacement`：控制样本是否可重复使用
- `method`：`"min"`（最近）、`"random"`（阈值内随机）或
  `"optimal"`（最小成本的一对一分配）
- `distance`：`"probability"` 或 `"logit"` 倾向得分距离，后者是常见的 logit caliper
- `exact`：要求匹配对中取值完全相同的列

## 实践建议

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
