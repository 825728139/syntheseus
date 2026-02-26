"""
FastAPI 服务器用于向前端提供逆合成搜索结果 API

提供以下端点：
- GET /api/routes: 列出所有可用的路由
- GET /api/route/{route_idx}: 获取指定路由的详细信息
- GET /api/graph: 获取完整图的 JSON 数据
- GET /: 静态前端页面
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 创建 FastAPI 应用
app = FastAPI(
    title="Retro Synthesis API",
    description="逆合成搜索结果 API",
    version="1.0.0"
)

# 添加 CORS 中间件（允许前端跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 结果目录（必须是 Path 对象才能使用 .glob() 方法）
RESULTS_DIR = Path("/home/liwenlong/retro_mcts_results/SimpRetro_2026-02-26T13:53:01")

# 挂载静态文件目录（用于前端页面）
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# Pydantic 模型用于 API 响应
class RouteSummary(BaseModel):
    """路由摘要信息"""
    index: int
    graph_type: str
    num_nodes: int
    num_edges: int


class NodeInfo(BaseModel):
    """节点信息"""
    id: str
    local_index: int
    type: str
    has_solution: bool
    depth: int
    smiles_list: List[str] | None = None
    smiles: str | None = None
    is_purchasable: bool | None = None
    data: Dict[str, Any] = {}


class EdgeInfo(BaseModel):
    """边信息"""
    source_id: str
    target_id: str
    source_index: int
    target_index: int
    reaction: Dict[str, Any] | None = None


class RouteDetail(BaseModel):
    """路由详细信息"""
    route_index: int
    graph_type: str
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    num_nodes: int
    num_edges: int


# API 端点
@app.get("/")
async def root():
    """根路径，返回 API 信息"""
    return {
        "message": "Retro Synthesis API",
        "version": "1.0.0",
        "endpoints": {
            "routes": "/api/routes",
            "route_detail": "/api/route/{route_idx}",
            "graph": "/api/graph"
        }
    }


@app.get("/api/routes", response_model=List[RouteSummary])
async def list_routes():
    """
    列出所有可用的路由

    返回所有路由的摘要信息，包括索引、图类型、节点数和边数。
    """
    routes = []

    # 搜索所有 route_*.json 文件
    for json_file in RESULTS_DIR.glob("route_*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                routes.append(RouteSummary(
                    index=data.get('route_index', 0),
                    graph_type=data.get('graph_type', 'Unknown'),
                    num_nodes=data.get('num_nodes', 0),
                    num_edges=data.get('num_edges', 0)
                ))
        except (json.JSONDecodeError, KeyError) as e:
            # 跳过无效的 JSON 文件
            continue

    # 按索引排序
    routes.sort(key=lambda r: r.index)
    return routes


@app.get("/api/route/{route_idx}")
async def get_route(route_idx: int):
    """
    获取指定路由的详细信息

    返回指定索引路由的完整 JSON 数据，包括所有节点和边的信息。
    """
    json_file = RESULTS_DIR / f"route_{route_idx}.json"
    # print(json_file)

    if not json_file.exists():
        raise HTTPException(status_code=404, detail=f"Route {route_idx} not found")

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            route_data = json.load(f)
        return route_data
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"Invalid JSON file for route {route_idx}")


@app.get("/api/graph")
async def get_graph():
    """
    获取完整图的 JSON 数据

    返回 graph_output.json 的内容，包含所有节点和边的完整图数据。
    """
    json_file = RESULTS_DIR / "graph_output.json"

    if not json_file.exists():
        raise HTTPException(status_code=404, detail="Graph output file not found")

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        return graph_data
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON file for graph output")


@app.get("/api/stats")
async def get_stats():
    """
    获取搜索统计信息

    返回 stats.json 的内容，包含搜索的统计数据。
    """
    json_file = RESULTS_DIR / "stats.json"

    if not json_file.exists():
        raise HTTPException(status_code=404, detail="Stats file not found")

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            stats_data = json.load(f)
        return stats_data
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON file for stats")


# 运行服务器
if __name__ == "__main__":
    import uvicorn

    # 打印配置信息
    print(f"Serving results from: {RESULTS_DIR}")
    print(f"API documentation available at: http://192.168.100.108:5000/docs")

    # 验证目录存在
    if not RESULTS_DIR.exists():
        print(f"WARNING: Results directory does not exist: {RESULTS_DIR}")
        print(f"Please run a search first to generate JSON files.")
    else:
        json_files = list(RESULTS_DIR.glob("route_*.json"))
        print(f"Found {len(json_files)} route JSON files")

    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=5000,
        reload=False  # 关闭 reload 以避免路径问题
    )
