# 逆合成搜索 (Retro Synthesis Search)

对目标产物分子执行逆合成路线规划。自动利用搜索图缓存续算，避免重复计算。

## 使用条件

当用户要求对某个分子进行逆合成分析、合成路线规划、retrosynthesis 时使用本技能。

## 执行流程

### 第一步：SMILES 校验

用 rdchiral 或 RDKit 校验用户提供的 SMILES 是否合法：

```bash
conda run -n syntheseus-full-mic python3 -c "
from rdkit import Chem
smiles = '<用户提供的SMILES>'
mol = Chem.MolFromSmiles(smiles)
if mol is None:
    print('INVALID')
else:
    print(Chem.MolToSmiles(mol))
"
```

- 校验失败：告知用户 SMILES 不合法，终止
- 校验成功：使用输出的标准规范式 SMILES 继续

### 第二步：执行逆合成搜索

调用逆合成搜索 tool，**始终使用 `save_graph: true`** 以支持续算。

**首轮请求**（expansion_time=40s）：

调用 `retrosynthesis_predict`，使用默认参数，入参格式：
```json
{
    "smiles": "<标准SMILES>",
    "backend": "mcts",
    "build_tree_options": {
      "expansion_time": 40,
      "max_branching": 10,
      "save_graph": true,
      "stop_on_first_solution": false,
      "expand_purchasable_target": true
    },
    "timeout": 120
}
```

### 第三步：判断结果并续算

检查响应中的 `num_routes_found`：

**有路线（≥1）** → 输出结果（见第五步）

**无路线 + 有 graph_id** → 用 `resume_from` 续算，不从头开始：

```json
{
    "smiles": "<标准SMILES>",
    "backend": "mcts",
    "build_tree_options": {
      "expansion_time": 30,
      "max_branching": 10,
      "save_graph": true,
      "resume_from": "<上一轮返回的graph_id>",
      "stop_on_first_solution": false,
      "expand_purchasable_target": true
    },
    "timeout": 120
}
```

**续算规则**：
- 每轮增加 30s 搜索时间
- 累计总搜索时间上限 300s（约 7-8 轮）
- 每轮都返回 `graph_id`，下一轮用 `resume_from` 接上
- 达到 300s 仍无路线 → 告知用户当前搜索状态，询问是否继续

**用户明确表示对结果不满意时**：继续续算，不受 300s 限制，每轮 30s 递增。

## 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `expansion_time` | 40（首轮） | 单轮搜索时间上限（秒），最小 30 |
| `max_branching` | 10 | 最多返回路线数 |
| `save_graph` | true | 始终开启，支持续算 |
| `resume_from` | null | 上一轮的 graph_id，续算时使用 |
| `stop_on_first_solution` | false | 找到第一个解不停止，继续搜索更优路线 |
| `expand_purchasable_target` | true | 即使目标在库存中也展开搜索 |
| `backend` | mcts | 搜索算法：mcts |
| `timeout` | 120 | HTTP 请求超时（秒），至少应大于 expansion_time 的值 20s |

## 注意事项

- `resume_from` 必须与原搜索使用相同的 `backend`
- 搜索图缓存超过 1 小时自动清理（`GRAPH_TTL_SECONDS` 环境变量可配置）
- expansion_time 最小 30s（服务端限制）
