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
    model_class=SimpRetro \
    model_dir=/path/to/model \
    time_limit_s=60 \
    search_algorithm=retro_star \
    num_routes_to_plot=10 \
    results_dir=results/ \
    mcts_config.max_expansion_depth=20 \
    expand_purchasable_target=True
```

### 交互模式

启用增量搜索，每30秒询问是否继续：

```bash
python search.py \
    inventory_smiles_file=emolecules.txt \
    search_target="NC1=Nc2ccc(F)cc2C2CCCC12" \
    model_class=SimpRetro \
    model_dir=/path/to/model \
    interactive_mode=True \
    increment_time_s=30 \
    max_continues=10 \
    search_algorithm=retro_star \
    num_routes_to_plot=10 \
    results_dir=results/
    mcts_config.max_expansion_depth=20 \
    expand_purchasable_target=True \
    resume_search=/retro_mcts_results/SimpRetro_2026-03-12T17:37:38
```

## 配置参数

### 新增参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `resume_search` | str | None | 是否启在目标pkl上继续扩展图 |
| `resume_search_time_s` | float | 30.0 | 每次扩展图的时间（秒） |

## 更新日志

### Version 1.0 (当前版本)

- 新增交互式增量搜索模式
- 新增 `interactive_mode`, `increment_time_s`, `max_continues` 配置参数
- 新增 `RouteTracker` 类用于路径追踪
- 新增 `print_interim_stats()` 函数用于中间结果输出
- 新增 `extract_and_plot_routes()` 函数用于增量路径提取
- 保持完全向后兼容性
