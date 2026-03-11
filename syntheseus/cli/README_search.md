# search.py 配置参数完整总结

## 配置类层次结构

```
SearchConfig
├── BackwardModelConfig (反应模型配置)
├── BaseSearchConfig (基础搜索配置)
│   └── SearchAlgorithmConfig (算法配置)
│       ├── RetroStarConfig
│       ├── MCTSConfig
│       └── PDVNConfig
```

---

## 一、BackwardModelConfig - 反应模型配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_dir` | str | MISSING | 模型文件目录 |
| `model_kwargs` | dict | {} | 传递给模型的额外参数 |
| `model_class` | BackwardModelClass | MISSING | 模型类选择（SimpRetro/LocalRetro/ChemFormer等） |

---

## 二、BaseSearchConfig - 基础搜索配置

### 目标分子配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `search_target` | str | MISSING | 单个目标分子的 SMILES |
| `search_targets_file` | str | MISSING | 包含多个目标分子的文件路径 |

**注意**：`search_target` 和 `search_targets_file` 必须且只能提供一个。

### 库存配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `inventory_smiles_file` | str | MISSING | 可购买分子的 SMILES 文件路径 |

### 结果目录配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `results_dir` | str | "." | 保存结果的目录 |
| `append_timestamp_to_dir` | bool | True | 是否在目录名后追加时间戳 |

### GPU 配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_gpu` | bool | True | 是否使用 GPU |
| `canonicalize_inventory` | bool | False | 是否规范化库存 SMILES |

### 反应模型配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_top_results` | int | 50 | 请求的顶级结果数量 |
| `reaction_model_use_cache` | bool | True | 是否缓存反应模型结果 |

### 保存配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `save_graph` | bool | True | 是否保存完整反应图 |
| `num_routes_to_plot` | int | 5 | 提取和绘制的路径数量 |

---

## 三、SearchAlgorithmConfig - 搜索算法配置

### 通用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `search_algorithm` | SearchAlgorithmClass | `retro_star` | 搜索算法选择（retro_star/mcts/pdvn） |

### 搜索限制

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `time_limit_s` | float | 600 | 最大搜索时间（秒） |
| `limit_reaction_model_calls` | int | 1,000,000 | 反应模型最大调用次数 |
| `limit_iterations` | int | 1,000,000 | 最大迭代次数 |
| `limit_graph_nodes` | int | ∞ | 图中最大节点数 |
| `prevent_repeat_mol_in_trees` | bool | True | 防止树中重复分子 |
| `stop_on_first_solution` | bool | False | 找到第一个解后是否停止 |
| `expand_purchasable_target` | **bool** | **True** | **是否扩展可购买的目标分子** |
[chemTools/retro_syn/syntheseus/syntheseus/search/mol_inventory.py:70] def is_purchasable

---

## 四、RetroStarConfig - Retro* 算法配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_expansion_depth` | int | 10 | 最大扩展深度 |
| `value_function_class` | str | "ConstantNodeEvaluator" | 价值函数类 |
| `value_function_kwargs` | dict | {"constant": 0.0} | 价值函数参数 |
| `and_node_cost_fn_class` | str | "ReactionModelLogProbCost" | AndNode 成本函数类 |
| `and_node_cost_fn_kwargs` | dict | {} | 成本函数参数 |

---

## 五、MCTSConfig - MCTS 算法配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_expansion_depth` | int | 20 | 最大扩展深度 |
| `value_function_class` | str | "ConstantNodeEvaluator" | 价值函数类 |
| `value_function_kwargs` | dict | {"constant": 0.5} | 价值函数参数 |
| `reward_function_class` | str | "HasSolutionValueFunction" | 奖励函数类 |
| `reward_function_kwargs` | dict | {} | 奖励函数参数 |
| `policy_class` | str | "ReactionModelProbPolicy" | 策略函数类 |
| `policy_kwargs` | dict | {} | 策略函数参数 |
| `bound_constant` | float | 1.0 | P-UCB 边界常数 |
| `bound_function_class` | str | "pucb_bound" | 边界函数类 |

---

## 六、PDVNConfig - PDVN 算法配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_expansion_depth` | int | 10 | 最大扩展深度 |
| `value_function_syn_class` | str | "ConstantNodeEvaluator" | 价值函数（合成路径）类 |
| `value_function_syn_kwargs` | dict | {"constant": 0.5} | 价值函数（合成路径）参数 |
| `value_function_cost_class` | str | "ConstantNodeEvaluator" | 价值函数（成本）类 |
| `value_function_cost_kwargs` | dict | {"constant": 0.0} | 价值函数（成本）参数 |
| `and_node_cost_fn_class` | str | "ConstantNodeEvaluator" | AndNode 成本函数类 |
| `and_node_cost_fn_kwargs` | dict | {"constant": 0.1} | 成本函数参数 |
| `policy_class` | str | "ReactionModelProbPolicy" | 策略函数类 |
| `policy_kwargs` | dict | {} | 策略函数参数 |
| `c_dead` | float | 5.0 | 死亡惩罚常数 |
| `bound_constant` | float | 1e2 | P-UCB 边界常数 |
| `bound_function_class` | str | "pucb_bound" | 边界函数类 |

---

## 七、交互式搜索模式配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `interactive_mode` | bool | False | 是否启用交互式增量搜索 |
| `increment_time_s` | float | 30.0 | 每次增量搜索的时间（秒） |
| `max_continues` | int | 10 | 最大继续提示次数 |

**交互模式说明**：
- 启用后，搜索会分批进行
- 每批搜索 `increment_time_s` 秒
- 用户可以选择是否继续下一批搜索
- 最多进行 `max_continues` 批

---

## 八、关键配置说明

### expand_purchasable_target

**作用**：控制是否对可购买的目标分子进行逆合成搜索

| 值 | 行为 |
|-----|------|
| `True` | 强制进行逆合成搜索，即使目标分子可购买 |
| `False` | 如果目标可购买，跳过搜索（no-op） |

**您已修改为 `True`**：这意味着即使目标分子在库存列表中，也会进行逆合成分析。

### stop_on_first_solution

| 值 | 行为 |
|-----|------|
| `True` | 找到第一个解后立即停止 |
| `False` | 继续搜索直到达到时间/迭代限制 |

### max_expansion_depth

控制搜索树的最大深度（即逆合成步骤数）：

| 算法 | 默认值 |
|------|--------|
| Retro* | 10 |
| MCTS | 20 |
| PDVN | 10 |

---

## 九、常用配置组合示例

### 快速原型（快速验证）
```python
search_algorithm = "mcts"
time_limit_s = 60
expand_purchasable_target = True
num_routes_to_plot = 3
```

### 深度搜索（寻找最优解）
```python
search_algorithm = "retro_star"
time_limit_s = 3600
max_expansion_depth = 15
stop_on_first_solution = False
```

### 交互式探索
```python
interactive_mode = True
increment_time_s = 30
max_continues = 20
```

---

## 十、关键文件位置

| 配置类 | 文件 |
|--------|------|
| 所有配置类 | `/home/liwenlong/chemTools/retro_syn/syntheseus/syntheseus/cli/search.py` |
| 反应模型配置 | `/home/liwenlong/chemTools/retro_syn/syntheseus/syntheseus/reaction_prediction/inference/config.py` |
| 节点评估器 | `/home/liwenlong/chemTools/retro_syn/syntheseus/syntheseus/search/node_evaluation/` |
| 算法实现 | `/home/liwenlong/chemTools/retro_syn/syntheseus/syntheseus/search/algorithms/` |
