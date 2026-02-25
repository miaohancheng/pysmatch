<!-- AUTO-GENERATED: DO NOT EDIT. -->
<!-- Source: ../../../README_CHINESE.md -->

# 常见问题

## 何时用 `linear`、`tree`、`knn`？

- `linear`：默认首选，解释性强，稳健。
- `tree`：适合非线性关系和复杂特征结构。
- `knn`：适合做局部结构对照和敏感性比较。

## 模型准确率越高越好吗？

不一定。准确率过高有时意味着组间重叠不足，反而会降低可匹配性。平衡性指标比单纯准确率更关键。

## 过采样还是欠采样怎么选？

- `over`：通常保留更多多数类信息，建议先用。
- `under`：训练更快，适合作为稳健性对照。

## 如何保证可复现？

- 固定随机种子（`np.random.seed(...)`）
- 固定依赖版本
- 记录完整的匹配参数与模型参数

## 延伸阅读

- Sekhon, J. S. (2011), *Multivariate and propensity score matching software with automated balance optimization: The Matching package for R*. Journal of Statistical Software, 42(7), 1-52. [Link](https://www.jstatsoft.org/article/view/v042i07)
- Rosenbaum, P. R., & Rubin, D. B. (1983), *The central role of the propensity score in observational studies for causal effects*. Biometrika, 70(1), 41-55. [Link](https://stat.cmu.edu/~ryantibs/journalclub/rosenbaum_1983.pdf)

## 贡献

欢迎提交 issue 和 pull request。

## 许可证

`pysmatch` 使用 MIT License。
