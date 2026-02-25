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
- `method`：`"min"`（最近）或 `"random"`（阈值内随机）

## 实践建议

- 先用 `nmatches=1`、`replacement=False` 和中等阈值起步。
- 若保留率太低，逐步放宽 `threshold`。
- 若匹配后仍不平衡，适当收紧阈值或更换模型。
- 类别极度不平衡时，可将 `balance_strategy="under"` 作为稳健性对照。
