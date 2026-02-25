<!-- AUTO-GENERATED: DO NOT EDIT. -->
<!-- Source: ../../../README_CHINESE.md -->

# 故障排查

## `ValueError: numpy.dtype size changed`

通常是 NumPy/Pandas 二进制兼容性问题。

```bash
pip install --upgrade --force-reinstall "numpy>=1.26.4" "pandas>=2.1.4"
```

重装后请重启 Python 内核或会话。

## `Scores column not found`

调用顺序应为：先拟合打分模型，再预测分数，再匹配。

```python
matcher.fit_scores(...)
matcher.predict_scores()
matcher.match(...)
```

## 读取数据报 `FileNotFoundError`

优先使用仓库相对路径：

```python
pd.read_csv("misc/loan.csv")
```

## 匹配结果为空

常见原因是阈值过严或组间重叠不足。

- 适度增大 `threshold`
- 更换 `model_type`
- 先用 `plot_scores()` 检查组间得分重叠

## Notebook 内核不可用

若 notebook 的 kernelspec 不存在，请切换到可用内核（如 `python3`）后重跑。
