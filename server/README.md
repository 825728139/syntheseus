# Retro Synthesis API Server

逆合成规划搜索服务，基于 FastAPI + 进程池，预热加载模型和分子库存。

## 启动

```bash
conda activate syntheseus-full-mic
cd /home/liwenlong/chemTools/retro_syn/syntheseus
uvicorn server.server:app --host 0.0.0.0 --port 5000
# 或直接运行
python server/server.py
```

## API 接口

### 阻塞式搜索

```
POST /api/retrosynthesis/search
```

等待搜索完成后返回结果。适合短任务（`expansion_time` 较小）。

### 异步搜索

```
POST /api/retrosynthesis/search/async   → 返回 task_id
GET  /api/retrosynthesis/status/{task_id} → 轮询状态/结果
```

立即返回，通过轮询获取结果。适合长任务（`expansion_time` 较大）。

### 健康检查

```
GET /health
```

## 请求参数

```json
{
  "smiles": "目标分子 SMILES（必填）",
  "backend": "mcts",
  "build_tree_options": {
    "expansion_time": 30.0,
    "max_branching": 50,
    "max_iterations": 0,
    "expand_purchasable_target": true,
    "stop_on_first_solution": false,
    "save_graph": false,
    "resume_from": null
  }
}
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `smiles` | string | (必填) | 目标分子 SMILES |
| `backend` | string | `"mcts"` | 搜索算法：`mcts` / `retro_star` / `pdvn` |
| `expansion_time` | float | `30.0` | 搜索时间上限（秒） |
| `max_branching` | int | `50` | 最多返回路线数 |
| `max_iterations` | int | `0` | 迭代次数上限，0 = 不限制 |
| `expand_purchasable_target` | bool | `true` | 即使目标在库存中也展开搜索 |
| `stop_on_first_solution` | bool | `false` | 找到第一个解即停止 |
| `save_graph` | bool | `false` | 缓存搜索图（用于续算） |
| `resume_from` | string | `null` | 从指定 graph_id 续算 |

## 搜索图缓存与续算

首次搜索时设置 `save_graph: true`，响应中会返回 `graph_id`：

```json
{
  "target_smiles": "...",
  "routes": [...],
  "graph_id": "a1b2c3d4e5f6"
}
```

续算时传入 `resume_from`：

```json
{
  "smiles": "...",
  "build_tree_options": {
    "resume_from": "a1b2c3d4e5f6",
    "expansion_time": 120
  }
}
```

搜索图保存在 `GRAPH_DIR`（默认 `./search_graphs/`），任务过期后自动清理。

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `INVENTORY_PATH` | `./emolecules.txt` | 分子库存文件路径 |
| `CANONICALIZE_INVENTORY` | `false` | 是否标准化库存 SMILES |
| `MODEL_DIR` | (硬编码路径) | SimpRetro 模型目录 |
| `USE_GPU` | `true` | 是否使用 GPU |
| `NUM_TOP_RESULTS` | `50` | 反应模型返回结果数 |
| `MAX_WORKERS` | `2` | 进程池 worker 数量 |
| `GRAPH_DIR` | `./search_graphs` | 搜索图缓存目录 |

## 文件结构

```
server/
├── server.py     # FastAPI 应用、进程池、缓存、任务管理
├── schemas.py    # Pydantic 请求/响应模型
├── worker.py     # Worker 进程：模型加载、搜索执行
├── engine.py     # 搜索引擎（单进程版，供直接调用）
└── __init__.py
```

## 缓存策略

- **请求缓存**：LRU，最多 100 条，永不过期。相同参数的请求直接返回缓存结果。
- **Graph 缓存**：`save_graph=true` 时保存搜索图 pkl 文件，TaskStore 过期（1h）时自动删除。
