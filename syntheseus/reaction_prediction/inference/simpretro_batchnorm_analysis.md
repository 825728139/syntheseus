# SimpRetro 预测逻辑分析报告

## 问题描述

当删除 [simpretro.py:148](syntheseus/reaction_prediction/inference/simpretro.py#L148) 中的一个在 [simpretro.py:216](syntheseus/reaction_prediction/inference/simpretro.py#L216) 中匹配的 SMARTS 模板后，[simpretro.py:255](syntheseus/reaction_prediction/inference/simpretro.py#L255) 中 `pred` 的所有模板预测值都会发生变化。

## 第 255 行的预测逻辑

```python
pred = self.filter(data).squeeze().cpu().numpy()
```

### 数据流向分析

1. **模板匹配阶段** ([simpretro.py:213-241](syntheseus/reaction_prediction/inference/simpretro.py#L213-L241))
   - 遍历所有模板，用 `rdchiralRun` 检查哪些模板能匹配当前产物
   - 匹配成功的模板索引存入 `valid_template_id` 列表

2. **神经网络过滤阶段** ([simpretro.py:244-255](syntheseus/reaction_prediction/inference/simpretro.py#L244-L255))
   ```python
   valid_temp_fps = self.template_fps[valid_template_id]  # 提取有效模板的指纹
   p_fp = smiles_to_fingerprint(x.smiles)  # 计算产物指纹
   data = torch.tensor(
       np.concatenate(
           [valid_temp_fps.squeeze(), np.repeat(p_fp, len(valid_temp_fps), axis=0)],
           axis=1,
       ),
       dtype=torch.float32,
   )
   pred = self.filter(data).squeeze().cpu().numpy()
   ```

   - `data` 形状：`(len(valid_temp_fps), 8192)`
   - 每一行是一个「模板指纹 (6144维) + 产物指纹 (2048维)」的拼接

3. **神经网络结构** ([model.py:78-92](syntheseus/reaction_prediction/fast_filter/model.py#L78-L92))
   ```python
   class Net_orig(nn.Module):
       def __init__(self):
           super(Net_orig, self).__init__()
           self.fc1 = nn.Linear(2048 * 4, 2048)
           self.fc2 = nn.Linear(2048, 1)
           self.bn = nn.BatchNorm1d(2048)  # ← 关键问题！

       def forward(self, x):
           x = torch.relu(self.fc1(x))
           x = self.bn(x)
           x = self.fc2(x)
           x = torch.sigmoid(x)
           return x
   ```

## 核心问题：缺少 `model.eval()`

在 [simpretro.py:176-183](syntheseus/reaction_prediction/inference/simpretro.py#L176-L183) 加载模型时：

```python
self.filter = Net_orig()
# 加载权重...
self.filter.load_state_dict(...)
# ❌ 没有调用 self.filter.eval()
```

**模型处于训练模式而非评估模式**。

## 为什么删除一个模板会影响所有预测值

### BatchNorm 在训练模式下的行为

训练模式下，`BatchNorm1d` 使用**当前批次**的统计量进行归一化：

```
normalized_x = (x - batch_mean) / sqrt(batch_var + eps)
```

### 变化链条

| 状态 | valid_temp_fps 数量 | batch_mean | batch_var | 归一化结果 |
|------|-------------------|------------|-----------|-----------|
| 删除前 | N | mean_N | var_N | x'_N = (x - mean_N) / sqrt(var_N) |
| 删除后 | N-1 | mean_{N-1} | var_{N-1} | x'_{N-1} = (x - mean_{N-1}) / sqrt(var_{N-1}) |

由于 **batch_mean 和 batch_var 都是基于整个批次计算的**，删除任何一个样本都会改变这些统计量，进而影响批次中**所有样本**的归一化结果。

### 数学解释

假设批次中有 N 个样本，第一个全连接层输出为 `h_1, h_2, ..., h_N`：

```
batch_mean = (h_1 + h_2 + ... + h_N) / N
batch_var = sum((h_i - batch_mean)^2) / N
```

删除一个样本后，新的统计量为：

```
batch_mean' = (h_1 + h_2 + ... + h_{N-1}) / (N-1) ≠ batch_mean
batch_var' = sum((h_i - batch_mean')^2) / (N-1) ≠ batch_var
```

因此，剩余所有样本的归一化值都会改变。

## 解决方案

在 [simpretro.py:183](syntheseus/reaction_prediction/inference/simpretro.py#L183) 后添加：

```python
self.filter.eval()
```

### eval() 模式下的行为

```python
# eval 模式使用训练时学习到的固定统计量
normalized_x = (x - running_mean) / sqrt(running_var + eps)
```

`running_mean` 和 `running_var` 是模型在训练期间累积的移动平均值，**不受当前 batch size 影响**。

## 验证方法

1. 添加 `self.filter.eval()` 后重新运行
2. 对比删除模板前后的预测值
3. 未删除的模板预测值应保持不变

## 总结

| 问题 | 根因 | 解决方案 |
|------|------|----------|
| 删除一个模板导致所有预测值变化 | 模型处于训练模式，BatchNorm 使用当前批次统计量 | 添加 `self.filter.eval()` |
