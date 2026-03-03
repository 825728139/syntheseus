"""
FastAPI 服务器用于向前端提供逆合成搜索结果 API

提供以下端点：
- GET /api/routes: 列出所有可用的路由
- GET /api/route/{route_idx}: 获取指定路由的详细信息
- GET /api/graph: 获取完整图的 JSON 数据
- GET /api/results/paths: 获取所有路径（Askcos 兼容格式）
- POST /api/search/start/: 启动逆合成搜索任务
- GET /api/search/status/{task_id}: 查询搜索任务状态
- GET /api/draw/: 绘制化学结构图片（Askcos 兼容）
- POST /api/rdkit/validate/: 验证 SMILES 语法
- POST /api/rdkit/canonicalize/: 标准化 SMILES
- GET /: 静态前端页面
"""
from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
import io

# 导入数据格式转换模块
from data_adapter import convert_to_askcos_format

# RDKit 绘图模块（仅在可用时导入）
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    Chem = None
    Draw = None
    rdMolDraw2D = None

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


# 搜索任务存储
search_tasks = {}
task_counter = 0
search_tasks_lock = threading.Lock()


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
            "graph": "/api/graph",
            "stats": "/api/stats",
            "paths_askcos": "/api/results/paths",
            "search_start": "/api/search/start/",
            "search_status": "/api/search/status/{task_id}",
            "draw": "/api/draw/",
            "rdkit_validate": "/api/rdkit/validate/",
            "rdkit_canonicalize": "/api/rdkit/canonicalize/",
            "template_sets": "/api/template/sets/"
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


@app.post("/api/rdkit/validate/")
async def validate_smiles(request: dict):
    """
    验证 SMILES 语法（Askcos 兼容端点）

    Args:
        request: 包含 "smiles" 字段的字典

    Returns:
        包含验证结果的字典
    """
    try:
        from rdkit import Chem
    except ImportError:
        # 如果 RDKit 不可用，返回简单验证
        smiles = request.get("smiles", "")
        return {
            "correct_syntax": bool(smiles and len(smiles) > 0),
            "valid_chem_name": bool(smiles and len(smiles) > 0),  # 简化处理
            "smiles": smiles
        }

    smiles = request.get("smiles", "")
    try:
        mol = Chem.MolFromSmiles(smiles)
        return {
            "correct_syntax": mol is not None,
            "valid_chem_name": mol is not None,  # 简化处理，实际应该检查化学名称
            "smiles": smiles
        }
    except Exception:
        return {"correct_syntax": False, "valid_chem_name": False, "smiles": smiles}


@app.post("/api/rdkit/canonicalize/")
async def canonicalize_smiles(request: dict):
    """
    标准化 SMILES（Askcos 兼容端点）

    Args:
        request: 包含 "smiles" 字段的字典

    Returns:
        包含标准化 SMILES 的字典
    """
    try:
        from rdkit import Chem
    except ImportError:
        # 如果 RDKit 不可用，返回原始 SMILES
        smiles = request.get("smiles", "")
        return {"smiles": smiles}

    smiles = request.get("smiles", "")
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            canonical = Chem.MolToSmiles(mol)
            return {"smiles": canonical}
    except Exception:
        pass
    return {"smiles": smiles}


@app.post("/api/search/start/")
async def start_search(request: dict):
    """
    启动逆合成搜索任务

    接收 SMILES，调用 search.py 进行搜索

    Args:
        request: 包含 "smiles" 等字段的字典

    Returns:
        包含 task_id 的任务信息
    """
    global task_counter

    smiles = request.get("smiles", "").strip()
    if not smiles:
        raise HTTPException(status_code=400, detail="SMILES is required")

    # 打印 SMILES（调试用）
    print(f"[SEARCH] Received SMILES: {smiles}")

    # 获取可选参数
    time_limit = request.get("time_limit_s", 30)
    num_routes = request.get("num_routes", 50)
    algorithm = request.get("search_algorithm", "mcts")

    # 获取 task_id（线程安全）
    with search_tasks_lock:
        task_id = f"task_{task_counter}"
        task_counter += 1

    # 构建命令行参数
    argv = [
        "inventory_smiles_file=/home/liwenlong/chemTools/retro_syn/syntheseus/emolecules.txt",
        f"search_target={smiles}",
        "model_class=SimpRetro",
        "model_dir=/home/liwenlong/chemTools/retro_syn/syntheseus/syntheseus/SimpRetro_templates copy.json",
        f"time_limit_s={time_limit}",
        f"search_algorithm={algorithm}",
        "results_dir=retro_mcts_results/",
        "use_gpu=False",
        f"num_routes_to_plot={num_routes}",
        "expand_purchasable_target=True",
    ]

    # 创建结果目录（带时间戳）
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    results_dir = Path(f"/home/liwenlong/retro_mcts_results/SimpRetro_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # 更新 argv 中的 results_dir
    argv = [arg if not arg.startswith("results_dir=") else f"results_dir={results_dir}" for arg in argv]

    # 保存任务信息
    with search_tasks_lock:
        search_tasks[task_id] = {
            "status": "running",
            "smiles": smiles,
            "results_dir": str(results_dir),
            "start_time": datetime.now().isoformat(),
            "argv": argv
        }

    # 在后台运行搜索
    def run_search():
        try:
            print(f"[SEARCH] Running search.py with argv: {argv}")
            # 导入并运行主搜索函数
            from syntheseus.cli.search import main as search_main
            import sys

            # 保存原始 sys.argv
            original_argv = sys.argv

            # 设置新的 argv
            sys.argv = ["search.py"] + argv

            try:
                # 运行搜索
                search_main()
                with search_tasks_lock:
                    search_tasks[task_id]["status"] = "completed"
                    search_tasks[task_id]["end_time"] = datetime.now().isoformat()
                    # 查找实际的结果目录（search.py 会创建带时间戳的子目录）
                    parent_dir = Path(search_tasks[task_id]["results_dir"])
                    # 查找包含 route_*.json 的子目录
                    actual_results_dir = None
                    for subdir in parent_dir.iterdir():
                        if subdir.is_dir() and list(subdir.glob("route_*.json")):
                            actual_results_dir = str(subdir)
                            break
                    if actual_results_dir:
                        search_tasks[task_id]["results_dir"] = actual_results_dir
                        print(f"[SEARCH] Actual results dir: {actual_results_dir}")
                print(f"[SEARCH] Task {task_id} completed")
            except Exception as e:
                with search_tasks_lock:
                    search_tasks[task_id]["status"] = "failed"
                    search_tasks[task_id]["error"] = str(e)
                print(f"[SEARCH] Task {task_id} failed: {e}")
            finally:
                # 恢复原始 sys.argv
                sys.argv = original_argv

        except Exception as e:
            print(f"[SEARCH] Failed to import search module: {e}")
            with search_tasks_lock:
                search_tasks[task_id]["status"] = "failed"
                search_tasks[task_id]["error"] = str(e)

    # 在后台线程运行
    thread = threading.Thread(target=run_search)
    thread.daemon = True
    thread.start()

    return {
        "task_id": task_id,
        "status": "running",
        "message": f"Search started for SMILES: {smiles}"
    }


@app.get("/api/search/status/{task_id}")
async def get_search_status(task_id: str):
    """
    查询搜索任务状态

    Args:
        task_id: 任务 ID

    Returns:
        包含任务状态的字典
    """
    with search_tasks_lock:
        if task_id not in search_tasks:
            raise HTTPException(status_code=404, detail="Task not found")

        task = search_tasks[task_id].copy()

    # 如果任务完成，返回结果路径
    result = {
        "task_id": task_id,
        "status": task["status"],
        "smiles": task["smiles"]
    }

    if task["status"] == "completed":
        result["results_dir"] = task["results_dir"]
        # 检查生成的文件
        results_dir = Path(task["results_dir"])
        route_files = list(results_dir.glob("route_*.json"))
        result["num_routes"] = len(route_files)

    elif task["status"] == "failed":
        result["error"] = task.get("error", "Unknown error")

    return result


@app.get("/api/results/paths")
async def get_all_paths(results_dir: str = Query(None, description="结果目录路径")):
    """
    获取所有路径（Askcos 兼容格式）

    返回所有路径的 Askcos 兼容格式数据，包括节点和边的信息。
    """
    routes = []

    # 使用指定的 results_dir 或默认的 RESULTS_DIR
    base_dir = Path(results_dir) if results_dir else RESULTS_DIR
    print(base_dir)
    # 搜索所有 route_*.json 文件
    for json_file in base_dir.glob("route_*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 转换为 Askcos 格式
                askcos_route = convert_to_askcos_format(data)
                routes.append(askcos_route)
        except (json.JSONDecodeError, KeyError):
            # 跳过无效的 JSON 文件
            continue

    # 按路径索引排序
    routes.sort(key=lambda r: int(r["id"].split("_")[1]) if "_" in r["id"] else 0)
    print('='*8)
    # print(routes)

    # 从第一条路径中提取 target_smiles（depth=0 的节点）
    target_smiles = ""
    if routes:
        # 找到 depth=0 的化学节点作为目标分子
        for node in routes[0].get("nodes", []):
            if node.get("type") == "chemical" and node.get("depth") == 0:
                target_smiles = node.get("smiles", "")
                break

    # 合并所有路径的节点，正确处理 terminal 属性
    merged_nodes = {}  # 节点 ID -> node dict
    merged_edges = []  # 所有边的列表

    for route in routes:
        # 合并节点
        for node in route.get("nodes", []):
            node_id = node["id"]
            if node_id in merged_nodes:
                # 节点已存在，需要合并属性
                existing = merged_nodes[node_id]

                # 对于化学节点，优先使用 terminal=true 的属性
                if node.get("type") == "chemical":
                    # 如果新节点的 terminal=true 而现有节点的 terminal=false，更新
                    if node.get("terminal") and not existing.get("terminal"):
                        existing["terminal"] = True
                        existing["depth"] = node.get("depth", existing.get("depth", 0))

                    # 更新 depth（取最大值）
                    if node.get("depth", 0) > existing.get("depth", 0):
                        existing["depth"] = node.get("depth", 0)
            else:
                # 新节点，直接添加
                merged_nodes[node_id] = node

        # 合并边
        merged_edges.extend(route.get("edges", []))

    # 创建合并后的路径
    merged_route = {
        "id": "merged",
        "nodes": list(merged_nodes.values()),
        "edges": merged_edges,
        "graph": {
            "depth": max((r.get("graph", {}).get("depth", 0) for r in routes)) if routes else 0,
            "num_reactions": sum((r.get("graph", {}).get("num_reactions", 0) for r in routes)) if routes else 0,
            "avg_plausibility": sum((r.get("graph", {}).get("avg_plausibility", 0) for r in routes)) / len(routes) if routes else 0,
            "min_plausibility": min((r.get("graph", {}).get("min_plausibility", float('inf')) for r in routes)) if routes else 0,
            "avg_score": sum((r.get("graph", {}).get("avg_score", 0) for r in routes)) / len(routes) if routes else 0,
            "min_score": min((r.get("graph", {}).get("min_score", float('inf')) for r in routes)) if routes else 0,
        }
    }

    print(f"[DEBUG] After merge: {len(merged_route['nodes'])} nodes, {len(merged_route['edges'])} edges")
    print("[DEBUG] Chemical nodes terminal status after merge:")
    for node in merged_route["nodes"]:
        if node.get("type") == "chemical":
            print(f"  {node['smiles']}: terminal={node.get('terminal')}, depth={node.get('depth')}")

    return {
        "paths": [merged_route],  # 返回单个合并后的路径
        "target_smiles": target_smiles,
        "stats": {
            "total_paths": 1,
            "total_chemicals": len([n for n in merged_route["nodes"] if n.get("type") == "chemical"]),
            "total_reactions": len([n for n in merged_route["nodes"] if n.get("type") == "reaction"])
        }
    }

async def _draw_molecule_smiles(smiles: str, svg: bool, transparent: bool, highlight: bool = False, size: int = 300):
    """绘制单个分子"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    if svg:
        # SVG 格式
        drawer = rdMolDraw2D.MolDraw2DSVG(size, size)
        if transparent:
            drawer.SetFontSize(0.8)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg_text = drawer.GetDrawingText()
        if isinstance(svg_text, bytes):
            svg_text = svg_text.decode('utf-8')
        return Response(content=svg_text, media_type="image/svg+xml")
    else:
        # PNG 格式
        img = Draw.MolToImage(mol, size=(size, size))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")


async def _draw_reaction_smiles(smiles: str, svg: bool, transparent: bool):
    """绘制反应"""
    # 分割反应物、试剂、产物
    parts = smiles.split(">")
    if len(parts) == 3:
        reactants_smiles, agents_smiles, products_smiles = parts
    elif len(parts) == 2:
        # 简单格式：reactants>>products
        reactants_smiles, products_smiles = parts
        agents_smiles = ""
    else:
        raise ValueError(f"Invalid reaction SMILES: {smiles}")

    # 解析分子
    reactants = [Chem.MolFromSmiles(s) for s in reactants_smiles.split(".") if s.strip()]
    products = [Chem.MolFromSmiles(s) for s in products_smiles.split(".") if s.strip()]

    if svg:
        # SVG 格式 - 简单拼接
        svg_parts = []

        # 反应物
        for mol in reactants:
            if mol:
                drawer = rdMolDraw2D.MolDraw2DSVG(200, 200)
                drawer.DrawMolecule(mol)
                drawer.FinishDrawing()
                svg_text = drawer.GetDrawingText()
                if isinstance(svg_text, bytes):
                    svg_text = svg_text.decode('utf-8')
                # 提取 SVG 内容（去掉 XML 声明和根标签）
                lines = svg_text.split('\n')
                content_lines = [l for l in lines if not l.startswith('<?xml') and '<svg' not in l and l.strip() and not l.startswith('</svg>')]
                svg_parts.append('\n'.join(content_lines))

        # 反应箭头
        svg_parts.append('<text x="25" y="100" text-anchor="middle" font-size="30">→</text>')

        # 产物
        for mol in products:
            if mol:
                drawer = rdMolDraw2D.MolDraw2DSVG(200, 200)
                drawer.DrawMolecule(mol)
                drawer.FinishDrawing()
                svg_text = drawer.GetDrawingText()
                if isinstance(svg_text, bytes):
                    svg_text = svg_text.decode('utf-8')
                lines = svg_text.split('\n')
                content_lines = [l for l in lines if not l.startswith('<?xml') and '<svg' not in l and l.strip() and not l.startswith('</svg>')]
                svg_parts.append('\n'.join(content_lines))

        # 组合 SVG
        combined_svg = f'<?xml version="1.0" encoding="UTF-8"?><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {200 * (len(reactants) + len(products) + 1)} 200">{"".join(svg_parts)}</svg>'
        return Response(content=combined_svg, media_type="image/svg+xml")
    else:
        # PNG 格式
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            # 如果 PIL 不可用，返回错误
            raise ValueError("PIL/Pillow is required for PNG output")

        images = [Draw.MolToImage(mol, size=(200, 200)) for mol in reactants if mol]

        # 添加箭头
        arrow = Image.new('RGB', (50, 200), (255, 255, 255))
        draw_img = ImageDraw.Draw(arrow)
        draw_img.text((15, 85), "→", fill=(0, 0, 0))
        images.append(arrow)

        images.extend([Draw.MolToImage(mol, size=(200, 200)) for mol in products if mol])

        # 水平拼接
        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)
        bg_color = (255, 255, 255) if not transparent else None
        if bg_color:
            combined = Image.new('RGB', (total_width, max_height), bg_color)
        else:
            combined = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for img in images:
            combined.paste(img, (x_offset, 0))
            x_offset += img.width

        buf = io.BytesIO()
        combined.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/api/draw/")
async def draw_structure(
    smiles: str = Query(..., description="SMILES 字符串"),
    svg: bool = Query(True, description="返回 SVG 格式"),
    transparent: bool = Query(False, description="透明背景"),
    highlight: bool = Query(False, description="高亮显示"),
    draw_map: bool = Query(False, description="显示原子映射"),
    input_type: str = Query(None, description="输入类型"),
    # 新增参数（可选，不影响现有功能）
    size: int = Query(300, description="图像大小（像素）"),
    annotate: bool = Query(False, description="添加注释（暂时忽略）"),
    ppg: float = Query(None, description="价格（暂时忽略）"),
    as_reactant: Optional[bool] = Query(None, description="作为反应物（暂时忽略）"),
    as_product: Optional[bool] = Query(None, description="作为产物（暂时忽略）"),
    reference: str = Query(None, description="参考 SMILES（暂时忽略）"),
):
    """
    绘制化学结构图片（Askcos 兼容端点）

    支持：
    - 分子 SMILES（如 CCO）
    - 反应 SMILES（如 CCO.CC(=O)O>>CCOC(=O)O）
    - 返回 SVG 或 PNG 格式
    """
    if not RDKIT_AVAILABLE:
        return Response(
            content='{"error": "RDKit is not available", "request": {"smiles": "' + smiles + '"}}',
            status_code=500,
            media_type="application/json"
        )

    try:
        # 判断是反应还是分子
        is_reaction = ">" in smiles or input_type == "reaction"

        if is_reaction:
            return await _draw_reaction_smiles(smiles, svg, transparent)
        else:
            return await _draw_molecule_smiles(smiles, svg, transparent, highlight, size)

    except Exception as e:
        return Response(
            content=f'{{"error": "Could not draw requested structure: {str(e)}", "request": {{"smiles": "{smiles}"}}}}',
            status_code=400,
            media_type="application/json"
        )

@app.get("/api/template/sets/")
async def get_template_sets():
    """
    获取模板集（Askcos 兼容端点 - 占位符）

    返回空的模板集列表，用于兼容 Askcos 前端。
    """
    return {
        "template_sets": [],
        "attributes": []
    }


# 运行服务器
if __name__ == "__main__":
    import uvicorn

    # 打印配置信息
    print(f"Serving results from: {RESULTS_DIR}")
    print(f"API documentation available at: http://192.168.100.108:5001/docs")

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
        port=5001,
        reload=False  # 关闭 reload 以避免路径问题
    )
