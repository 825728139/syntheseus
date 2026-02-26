# 置信度计算与路径提取分析

## 背景

在逆合成搜索中，搜索算法（如 MCTS、Retro*）会生成多条可能的合成路径。如何评估这些路径的可靠性，并按置信度排序提取最优路径，是一个关键问题。

## 一、Syntheseus 置信度计算框架

### 1.1 反应模型的置信度来源

Syntheseus 支持多种反应预测模型，每种模型的置信度计算方式不同：

| 模型 | 置信度类型 | 计算方式 | 文件位置 |
|------|-----------|---------|----------|
| **SimpRetro** | `probability` | 模板匹配评分 + 神经网络过滤，经 softmax 归一化 | `simpretro.py:275-281` |
| **LocalRetro** | `probability` | 模型输出的概率分数（0-1） | `local_retro.py:116-129` |
| **ChemFormer** | `log_probability`, `probability` | 模型的对数似然，exp() 转换为概率 | `chemformer.py:152-154` |

### 1.2 SimpRetro 模型的置信度计算详解

```python
# 文件: syntheseus/reaction_prediction/inference/simpretro.py

# 1. 多维度评分 (第 231-237 行)
score = 1 * (
    w1 * CDScore(p_mol, r.split("."))        # 复杂度差异评分 (w1=0.1)
    + w2 * ASScore(p_mol, canonical_r_dict, instock_list)  # 可用性评分 (w2=0.2)
    + w3 * rdscore                           # 环差异评分 (w3=0.5)
    + w4 * 1 / len(mapped_curr_results)      # 多样性评分 (w4=0)
)

# 2. Softmax 归一化 (第 275-281 行)
scores = [np.exp(s) for s in scores]  # 指数化
total = sum(scores)
scores = [s / total for s in scores]  # 归一化求和为1
```

**合理性分析**：
- **CDScore** (Complexity Difference): 优先选择原子数减少的反应，符合化学直觉
- **ASScore** (Availability Score): 优先选择可购买的原料，提高实用性
- **RDScore** (Ring Difference): 优先选择开环反应，降低合成难度
- **权重设计**: w3=0.5 最高，说明环结构变化是最重要的因素

### 1.3 统一的置信度处理框架

Syntheseus 提供了 `ReactionModelBasedEvaluator` 类来统一处理置信度：

```python
# 文件: syntheseus/search/node_evaluation/base.py (第 142-160 行)

def _evaluate_nodes(self, nodes, graph=None):
    # 1. 获取原始概率
    probs = np.asarray([self._get_probability(n, graph) for n in nodes])

    # 2. 裁剪（防止极端值）
    probs = np.clip(probs, a_min=1e-10, a_max=0.999)

    # 3. 温度缩放
    if self._return_log:
        outputs = np.log(probs) / self._temperature
        if self._normalize:
            outputs -= outputs.max()  # log_softmax
            outputs -= np.log(np.exp(outputs).sum())
    else:
        outputs = probs ** (1.0 / self._temperature)
        if self._normalize:
            outputs /= outputs.sum()

    return outputs.tolist()
```

---

## 二、路径置信度计算实现

### 2.1 核心思路

**路径置信度 = 路径中所有反应概率的乘积**

这是基于概率论的独立性假设：如果各个反应步骤相互独立，则整个路径的概率为各步骤概率的乘积。

**对数空间计算**（避免数值下溢）：
```
log(路径置信度) = sum(log(反应概率))
路径置信度 = exp(log(路径置信度))
```

### 2.2 实现代码

新增文件：`syntheseus/search/analysis/route_extraction.py`

#### 辅助函数：从路径中提取反应

```python
def _get_reactions_from_route(route, graph):
    """兼容 MolSetGraph 和 AndOrGraph 的反应提取"""
    reactions = []
    if isinstance(graph, AndOrGraph):
        # AndOrGraph: 反应存储在 AndNode 中
        for node in route:
            if isinstance(node, AndNode):
                reactions.append(node.reaction)
    elif isinstance(graph, MolSetGraph):
        # MolSetGraph: 反应存储在边上
        route_list = list(route)
        for i in range(len(route_list) - 1):
            u, v = route_list[i], route_list[i + 1]
            edge_data = graph._graph.get_edge_data(u, v)
            if "reaction" in edge_data:
                reactions.append(edge_data["reaction"])
    return reactions
```

#### 成本函数（用于优先级队列排序）

```python
def _route_neg_log_prob(nodes, graph):
    """负对数概率作为成本（越小=置信度越高）"""
    if not _route_has_solution(nodes, graph):
        return math.inf

    log_prob_sum = 0.0
    reactions = _get_reactions_from_route(nodes, graph)

    for reaction in reactions:
        if "log_probability" in reaction.metadata:
            log_prob_sum += reaction.metadata["log_probability"]
        elif "probability" in reaction.metadata:
            prob = reaction.metadata["probability"]
            if prob > 0:
                log_prob_sum += math.log(prob)
            else:
                return math.inf

    return -log_prob_sum  # 负值：越小越好
```

#### 主提取函数

```python
def iter_routes_confidence_order(
    graph, max_routes, max_time_s=math.inf,
    min_confidence=None, return_confidence=False
):
    """按置信度从高到低提取路径"""
    for cost, route in _iter_top_routes(
        graph=graph,
        cost_fn=_route_neg_log_prob,
        cost_lower_bound=_route_neg_log_prob_partial,
        max_routes=max_routes,
        max_time_s=max_time_s,
        yield_partial_routes=False,
    ):
        confidence = math.exp(-cost) if cost < 100 else 0.0

        if min_confidence is not None and confidence < min_confidence:
            break

        if return_confidence:
            yield confidence, route
        else:
            yield route
```

---

## 三、设计合理性与意义

### 3.1 设计合理性

#### 1. **兼容两种图结构**

Syntheseus 使用两种不同的图结构：

| 图类型 | 使用算法 | 反应存储位置 |
|--------|----------|-------------|
| MolSetGraph | MCTS | 边上 |
| AndOrGraph | Retro*, PDVN | AndNode 中 |

我的实现通过 `isinstance` 判断图类型，分别处理：
- **AndOrGraph**: 遍历节点，提取 AndNode.reaction
- **MolSetGraph**: 遍历边，提取 edge_data["reaction"]

#### 2. **对数空间计算**

直接计算多个小概率的乘积会导致数值下溢：
```
例如: 0.1 × 0.2 × 0.3 × 0.4 × 0.5 = 0.0012
随着路径增长，乘积会迅速接近0
```

使用对数空间：
```
log(0.0012) = log(0.1) + log(0.2) + ... = -6.725
```

#### 3. **负对数作为成本**

`_iter_top_routes` 函数使用最小堆（成本越低优先级越高）：
```
成本 = -log(置信度)
置信度越高 → log(置信度) 越接近0 → 负对数越小 → 优先级越高
```

### 3.2 应用意义

#### 1. **路径排序与筛选**

```python
# 按置信度提取前10条路径
routes = iter_routes_confidence_order(graph, max_routes=10)

# 设置最小置信度阈值
routes = iter_routes_confidence_order(
    graph, max_routes=100, min_confidence=0.01
)
```

#### 2. **多目标优化**

置信度可以与其他指标结合：
- **成本**: 路径的经济成本
- **步骤数**: 反应步骤的数量
- **置信度**: 路径的可靠性

可以设计 Pareto 最优解：在置信度足够高的情况下，选择成本最低的路径。

#### 3. **可解释性**

输出每条路径的置信度，让化学家判断：
```
Route 0: confidence = 0.123  (推荐)
Route 1: confidence = 0.045
Route 2: confidence = 0.012
```

---

## 四、思路来源

### 4.1 参考现有实现

我分析了 Syntheseus 现有的路径提取函数：

| 函数 | 排序依据 | 代码位置 |
|------|----------|----------|
| `iter_routes_time_order` | 路径创建时间 | route_extraction.py:225-249 |
| `iter_routes_cost_order` | 路径成本 | route_extraction.py:171-203 |

这两个函数都使用 `_iter_top_routes` 通用框架，只需要提供：
- `cost_fn`: 计算完整路径的成本
- `cost_lower_bound`: 计算部分路径的成本下界

### 4.2 概率论基础

路径置信度计算基于概率论的**独立性假设**：

对于事件序列 A → B → C → D：
```
P(A→B→C→D) = P(A→B) × P(B→C) × P(C→D)
```

这是马尔可夫链的基本假设，在逆合成规划中被广泛采用。

### 4.3 对数空间的数值稳定性

在机器学习中，对数概率是标准做法：

```python
# 交叉熵损失
loss = -sum(y * log(p))

# 语言模型概率
P(sentence) = product(P(word|context))  # 数值不稳定
log P(sentence) = sum(log P(word|context))  # 数值稳定
```

---

## 五、使用示例

### 5.1 基本使用

```python
from syntheseus.search.analysis.route_extraction import iter_routes_confidence_order

# 按置信度提取路径
routes = iter_routes_confidence_order(
    output_graph,
    max_routes=10,
    return_confidence=True
)

for route_idx, (confidence, route) in enumerate(routes):
    print(f"Route {route_idx}: confidence = {confidence:.6f}")
```

### 5.2 在 search.py 中集成

```python
# 替换原有的路径提取代码
if config.num_routes_to_plot > 0:
    from syntheseus.search.analysis.route_extraction import iter_routes_confidence_order

    routes = iter_routes_confidence_order(
        output_graph,
        max_routes=config.num_routes_to_plot,
        min_confidence=0.001,
        return_confidence=True
    )

    for route_idx, (confidence, route) in enumerate(routes):
        logger.info(f"Route {route_idx}: confidence = {confidence:.6f}")
        # 保存路径
        with open(results_dir / f"route_{route_idx}.pkl", "wb") as f_route:
            pickle.dump(route, f_route)
        # 绘制路径
        visualize_andor(
            graph=output_graph,
            filename=str(results_dir / f"route_{route_idx}.pdf"),
            nodes=route,
        )
```

---

## 六、总结

### 设计特点

1. **兼容性**: 同时支持 MolSetGraph (MCTS) 和 AndOrGraph (Retro*, PDVN)
2. **数值稳定**: 使用对数空间计算，避免数值下溢
3. **可扩展**: 可以轻松添加其他置信度计算方式
4. **高效**: 复用现有的 `_iter_top_routes` 框架

### 潜在改进

1. **贝叶斯修正**: 考虑路径长度对置信度的影响
2. **先验知识**: 融入化学反应的先验成功率
3. **集成学习**: 结合多个模型的置信度预测
4. **不确定性量化**: 输出置信度的置信区间

### 参考文献

- Chen, H. et al. (2020). Retro*: Learning to Retrosynthesize. *ICML*.
- Segler, M. H. S. et al. (2018). Planning chemical syntheses with deep neural networks and symbolic AI. *Nature*.
- Shi, K. et al. (2022). LocalRetro: A Local Framework for Retrosynthesis. *JACS Au*.

---

## 七、节点数据参数详解

Syntheseus 在搜索过程中会在节点的 `data` 字典中存储各种参数，用于算法决策和结果分析。

### 7.1 MolSetNode 数据（MCTS 算法）

#### 实际数据示例

```python
data = {
    'policy_score': 0.32014232749444343,
    'num_calls_rxn_model': 3,
    'num_calls_value_function': 2,
    'mcts_value': 1.0,
    'mcts_prev_reward': 1.0,
    'analysis_time': 7.432425,
    'first_solution_time': 7.432425
}
```

#### 参数详解

| 参数 | 值 | 含义 | 化学参考意义 |
|------|-----|------|-------------|
| `policy_score` | 0.320 | 反应策略分数 | 反应模型认为此反应有 32% 的成功率 |
| `mcts_value` | 1.0 | MCTS 价值估计 | 完美解：所有分子都可购买 |
| `mcts_prev_reward` | 1.0 | 最近一次奖励 | 最新评估确认路径可行 |
| `num_calls_rxn_model` | 3 | 反应模型调用次数 | 搜索进度/计算成本指标 |
| `num_calls_value_function` | 2 | 价值函数调用次数 | 路径评估次数 |
| `analysis_time` | 7.43s | 节点创建时间 | 此路径在搜索开始后 7.43 秒被发现 |
| `first_solution_time` | 7.43s | 首个解时间 | 7.43 秒内找到第一个解 |

**算法核心代码** (`mcts/base.py`):

```python
# P-UCB 边界函数 (第 93-106 行)
def pucb_bound(node, graph):
    policy_score = node.data["policy_score"]
    return policy_score * math.sqrt(parents[0].num_visit) / (1 + node.num_visit)

# 价值更新 (第 310-318 行)
def _update_value_from_reward(self, node, reward):
    if node.num_visit == 0:
        node.data["mcts_value"] = reward
    else:
        total_reward = reward + node.data["mcts_value"] * node.num_visit
        node.data["mcts_value"] = total_reward / (node.num_visit + 1)  # 移动平均
    node.data["mcts_prev_reward"] = reward
```

**奖励函数设计** (HasSolutionValueFunction):
- 如果节点所有分子都可购买 → 奖励 = 1.0
- 否则 → 奖励 = 0.0

---

### 7.2 AndNode 数据（Retro* 算法）

#### 实际数据示例

```python
data = {
    'num_calls_rxn_model': 1,
    'num_calls_value_function': 0,
    'retro_star_rxn_cost': 0.017146158834970514,
    'retro_star_min_cost': 0.017146158834970514,
    'retro_star_reaction_number': 0.017146158834970514,
    'retro_star_value': 0.017146158834970514,
    'analysis_time': 1.77051,
    'first_solution_time': 1.77051
}
```

#### 参数详解

| 参数 | 值 | 含义 | 化学参考意义 |
|------|-----|------|-------------|
| `retro_star_rxn_cost` | 0.017 | 单个反应成本 | 反应概率约 98.3%，高置信度反应 |
| `retro_star_min_cost` | 0.017 | 最小累积成本 | 从根到当前节点的最低成本路径 |
| `retro_star_reaction_number` | 0.017 | 反应数估计 | 考虑反应难度的等效步数 |
| `retro_star_value` | 0.017 | Retro* 值 | 搜索优先级：值越小优先级越高 |
| `num_calls_rxn_model` | 1 | 反应模型调用次数 | 搜索早期节点 |
| `num_calls_value_function` | 0 | 价值函数调用次数 | 尚未进行额外评估 |
| `analysis_time` | 1.77s | 节点创建时间 | 此路径在 1.77 秒被发现 |
| `first_solution_time` | 1.77s | 首个解时间 | 1.77 秒找到第一个解 |

**算法核心代码** (`retro_star.py`):

```python
# 最小成本更新 (第 184-205 行)
def min_cost_update(node, graph):
    if isinstance(node, AndNode):
        new_cost = node.data["retro_star_rxn_cost"] + sum(
            c.data["retro_star_min_cost"] for c in graph.successors(node)
        )

# 反应数更新 (第 208-240 行)
def reaction_number_update(node, graph):
    if isinstance(node, AndNode):
        new_rn = node.data["retro_star_rxn_cost"] + sum(
            c.data["retro_star_reaction_number"] for c in graph.successors(node)
        )

# Retro* 值更新 (第 243-296 行)
def retro_star_value_update(node, graph):
    if isinstance(node, AndNode):
        new_value = (
            parent.data["retro_star_value"]
            - parent.data["retro_star_reaction_number"]
            + node.data["retro_star_reaction_number"]
        )
```

---

### 7.3 Retro* 与 MCTS 参数对比

#### 核心差异

| 维度 | Retro* | MCTS |
|------|--------|------|
| **图结构** | AndOrGraph (双重节点) | MolSetGraph (单一节点) |
| **价值表示** | 成本 (Cost) | 奖励 (Reward) |
| **值域** | [0, ∞)，越低越好 | [0, 1]，越高越好 |
| **评分依据** | 反应对数概率 | 累积成功奖励 |
| **决策方式** | 确定性优先级队列 | P-UCB 概率探索 |
| **存储位置** | AndNode.reaction.metadata | 边上的 reaction 对象 |

#### 策略指导对比

```python
# Retro*：使用反应成本
retro_star_rxn_cost = -log(reaction_probability)  # 概率 98.3% → 成本 0.017

# MCTS：使用策略分数
policy_score = reaction_probability  # 直接使用概率，如 0.320
```

#### 优先级计算对比

```python
# Retro*：确定性排序
priority = retro_star_value  # 值越小，优先级越高

# MCTS：P-UCB 边界
priority = mcts_value - c * policy_score * sqrt(N_parent) / (1 + N_child)
```

#### 化学意义对比

| 方面 | Retro* | MCTS |
|------|--------|------|
| **反应难度量化** | 成本直接反映难度 | 隐含在奖励中 |
| **路径选择** | 选择最低成本路径 | 平衡探索与利用 |
| **收敛特性** | 单调改进（成本递减） | 随机波动（平均值收敛） |
| **适用场景** | 需要最优解 | 需要多样性 |

---

### 7.4 数值解读指南

#### 反应成本与概率的对应关系

| 成本 | 概率 | 化学解读 |
|------|------|----------|
| 0.01 | 99.0% | 极高置信度，常见反应 |
| 0.05 | 95.1% | 高置信度，可靠反应 |
| 0.10 | 90.5% | 中高置信度，常规反应 |
| 0.50 | 60.7% | 中等置信度，需要验证 |
| 1.00 | 36.8% | 低置信度，谨慎使用 |
| 2.00 | 13.5% | 极低置信度，高风险 |

**计算公式**:
```python
probability = exp(-retro_star_rxn_cost)
retro_star_rxn_cost = -log(probability)
```

#### MCTS 奖励解读

| mcts_value | 解的状态 | 化学解读 |
|-----------|----------|----------|
| 1.0 | 完全解决 | 所有分子都可购买，路径完整 |
| 0.5-1.0 | 部分解决 | 大部分分子可购买 |
| 0.0-0.5 | 探索中 | 路径不完整或验证中 |
| 0.0 | 无解 | 无法找到完整路径 |

---

### 7.5 关键代码位置

| 功能 | 文件路径 |
|------|----------|
| 节点数据定义 | `syntheseus/search/graph/node.py` |
| MCTS 核心算法 | `syntheseus/search/algorithms/mcts/base.py` |
| Retro* 核心算法 | `syntheseus/search/algorithms/best_first/retro_star.py` |
| P-UCB 边界 | `mcts/base.py:93-106` |
| 价值更新 | `mcts/base.py:310-318` |
| 成本更新 | `retro_star.py:184-205` |
| 反应数更新 | `retro_star.py:208-240` |
| Retro* 值更新 | `retro_star.py:243-296` |
