# search.py 交互式增量搜索功能文档

## 概述

`search.py` 现在支持**交互式增量搜索模式**，允许用户在搜索过程中分批次查看结果，并决定是否继续搜索。

### 功能特点

1. **增量搜索**：每次搜索固定时间（默认30秒）后输出结果
2. **状态保留**：后续搜索在已有搜索图的基础上继续扩展
3. **路径去重**：自动追踪已显示的路径，仅显示新发现的路径
4. **用户控制**：每次输出后询问用户是否继续搜索

## 使用方法

### 非交互模式（默认）

保持原有行为，一次性完成搜索：

```bash
python search.py \
    inventory_smiles_file=emolecules.txt \
    search_target="NC1=Nc2ccc(F)cc2C2CCCC12" \
    model_class=RetroKNN \
    model_dir=/path/to/model \
    time_limit_s=60 \
    search_algorithm=retro_star \
    num_routes_to_plot=10 \
    results_dir=results/
```

### 交互模式

启用增量搜索，每30秒询问是否继续：

```bash
python search.py \
    inventory_smiles_file=emolecules.txt \
    search_target="NC1=Nc2ccc(F)cc2C2CCCC12" \
    model_class=RetroKNN \
    model_dir=/path/to/model \
    interactive_mode=True \
    increment_time_s=30 \
    max_continues=10 \
    search_algorithm=retro_star \
    num_routes_to_plot=10 \
    results_dir=results/
```

## 配置参数

### 新增参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `interactive_mode` | bool | False | 是否启用交互模式 |
| `increment_time_s` | float | 30.0 | 每次增量搜索的时间（秒） |
| `max_continues` | int | 10 | 最大继续询问次数 |

### 参数说明

- **interactive_mode**: 设为 `True` 启用交互模式，`False` 保持原有行为
- **increment_time_s**: 每轮搜索的时间限制，建议根据任务复杂度调整（10-120秒）
- **max_continues**: 防止无限循环的保护机制，达到此次数后自动停止

## 输出格式

### 交互模式输出示例

```
============================================================
已搜索 30.0 秒 (第 1 轮)
============================================================
迭代轮次: 1
累计搜索时间: 30.0 秒
图中节点总数: 1523
反应模型调用次数: 45
目标分子: NC1=Nc2ccc(F)cc2C2CCCC12
目标在库存中: False
是否找到解: 是
已发现解的数量: 3
本轮新发现路径: 3 条

是否继续搜索 30 秒？(y/n):
```

### 输出文件

交互模式下，路径文件按迭代轮次命名：
- `route_0.json`, `route_0.pdf` - 第1轮发现的第1条路径
- `route_1.json`, `route_1.pdf` - 第1轮发现的第2条路径
- `route_2.json`, `route_2.pdf` - 第1轮发现的第3天路径
- ...
- `uds_askcos.json` - 第1轮发现的前num_routes_to_plot条路径集合（前端需求的json格式）

## 注意事项

### 1. 向后兼容性

- **默认行为不变**：`interactive_mode=False` 时，程序行为与之前完全相同
- **逐步迁移**：可以逐步测试交互模式，不影响现有工作流

### 2. 搜索状态管理

- **图状态保留**：搜索图（`output_graph`）在迭代间保持完整
- **节点防重复扩展**：已扩展节点的 `is_expanded` 标志防止重复计算
- **反应模型缓存**：缓存机制继续有效，避免重复调用

### 3. 性能考虑

| 方面 | 说明 |
|------|------|
| 内存使用 | 搜索图会持续增长，长时间搜索可能占用较多内存 |
| 计算效率 | 增量搜索不影响总计算量，仅改变输出时机 |
| 路径追踪 | `RouteTracker` 使用内存哈希，影响可忽略 |

### 4. 用户交互

- **键盘中断**：支持 `Ctrl+C` 优雅退出
- **自动保存**：即使中断，已完成的搜索结果也会保存

### 5. 限制

- **单目标优化**：交互模式仅优化单目标搜索体验
- **多目标搜索**：多目标搜索时，每个目标独立完成

## 技术实现细节

### 修改范围

仅修改单个文件：`syntheseus/cli/search.py`

| 项目 | 代码量 | 说明 |
|------|--------|------|
| 配置参数 | ~3 行 | BaseSearchConfig 新增字段 |
| 辅助类/函数 | ~140 行 | RouteTracker, print_interim_stats, extract_and_plot_routes |
| 搜索逻辑 | ~60 行 | run_from_config 中的增量搜索循环 |

### 核心机制

```python
# 增量搜索核心逻辑（简化版）
output_graph = None
route_tracker = RouteTracker()

for iteration in range(max_continues):
    # 设置本次时间限制
    alg.time_limit_s = increment_time_s

    # 首次或继续搜索
    if output_graph is None:
        output_graph, _ = alg.run_from_mol(target)
    else:
        _, _ = alg.run_from_graph(output_graph)

    # 输出统计和新路径
    print_interim_stats(...)
    extract_and_plot_routes(...)

    # 询问是否继续
    if not ask_user_to_continue():
        break
```

### 关键设计决策

1. **图保留策略**：保留完整图对象，而非序列化/反序列化
2. **路径去重**：基于节点对象ID的哈希，高效且准确
3. **时间控制**：临时覆盖 `alg.time_limit_s`，恢复原始值
4. **异常处理**：捕获 `KeyboardInterrupt` 确保数据安全

## 测试建议

### 功能验证

1. **非交互模式**：验证原有行为未受影响
   ```bash
   python search.py ... time_limit_s=60
   ```

2. **交互模式-单轮**：验证基本功能
   ```bash
   python search.py ... interactive_mode=True increment_time_s=10 max_continues=1
   ```

3. **交互模式-多轮**：验证状态保留
   ```bash
   python search.py ... interactive_mode=True increment_time_s=30 max_continues=3
   ```

### 正确性验证

对比一次性搜索 vs 增量搜索的结果一致性：

```bash
# 一次性搜索 90 秒
python search.py ... time_limit_s=90 results_dir=baseline/

# 增量搜索 3×30 秒
python search.py ... interactive_mode=True increment_time_s=30 max_continues=3 results_dir=incremental/

# 比较结果（节点数、解的数量等）
```

## 常见问题

### Q1: 交互模式会影响搜索结果吗？

**A**: 不会。增量搜索仅改变输出时机，最终结果与一次性搜索相同。

### Q2: 如何选择 `increment_time_s`？

**A**: 根据任务复杂度和耐心：
- 简单任务：10-30秒
- 中等任务：30-60秒
- 复杂任务：60-120秒

### Q3: 可以在多目标搜索中使用吗？

**A**: 可以，但交互模式会应用于每个目标，建议先用单目标测试。

### Q4: 如何在没有 TTY 的环境（如脚本）中使用？

**A**: 使用非交互模式（`interactive_mode=False`），或通过管道提供输入：
```bash
echo -e "y\ny\nn" | python search.py ... interactive_mode=True
```

### Q5: 路径去重是如何工作的？

**A**: `RouteTracker` 基于节点对象ID计算哈希，确保每条路径只显示一次。

## 更新日志

### Version 1.0 (当前版本)

- 新增交互式增量搜索模式
- 新增 `interactive_mode`, `increment_time_s`, `max_continues` 配置参数
- 新增 `RouteTracker` 类用于路径追踪
- 新增 `print_interim_stats()` 函数用于中间结果输出
- 新增 `extract_and_plot_routes()` 函数用于增量路径提取
- 保持完全向后兼容性
