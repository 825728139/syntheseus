"""
Syntheseus 到 Askcos 数据格式转换模块

将 Syntheseus 的路径数据格式转换为 Askcos-vue-nginx 前端可识别的格式。
"""
from typing import Dict, List, Any
from uuid import uuid4


def convert_to_askcos_format(route_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    将 Syntheseus 路径数据转换为 Askcos 兼容格式

    Syntheseus 格式:
    - nodes: MolSetNode (smiles_list, is_purchasable, depth)
    - edges: with reaction metadata on edge

    Askcos 格式:
    - nodes: chemical/reaction 分离
    - edges: simple connections

    Args:
        route_data: Syntheseus 路径数据

    Returns:
        Askcos 兼容格式的路径数据
    """
    edges = []
    node_id_map = {}  # Syntheseus ID -> Askcos ID (可以是单个 SMILES 或 SMILES 列表)

    # 使用字典存储已创建的化学节点，便于后续更新属性
    created_nodes = {}  # SMILES -> node dict

    # 处理 Syntheseus 节点
    for idx, node in enumerate(route_data["nodes"]):
        # 获取 SMILES 列表
        smiles = node.get("smiles")
        smiles_list = node.get("smiles_list", [])
        if smiles:
            # 将 smiles 放在列表开头，并去重
            smiles_list = [smiles] + [s for s in smiles_list if s != smiles]

        if not smiles_list:
            continue

        # 获取节点属性用于计算 terminal
        is_purchasable = node.get("is_purchasable", False)
        depth = node.get("depth", 0)
        terminal = is_purchasable and depth > 0

        # 为每个 SMILES 创建或更新化学节点
        node_smiles_ids = []  # 存储此 Syntheseus 节点的所有 SMILES ID
        for smiles in smiles_list:
            askcos_node_id = smiles

            if askcos_node_id in created_nodes:
                # 节点已存在，更新属性（优先使用更合适的值）
                existing = created_nodes[askcos_node_id]

                # 如果新节点的 terminal=true 而现有节点的 terminal=false，更新
                if terminal and not existing.get("terminal", False):
                    existing["terminal"] = terminal
                    existing["depth"] = depth

                # 更新 depth（取最大值）
                if depth > existing.get("depth", 0):
                    existing["depth"] = depth

                # 更新 ppg 属性（如果节点是可购买的）
                if is_purchasable:
                    existing["ppg"] = 1
            else:
                # 创建新节点
                askcos_node = {
                    "id": askcos_node_id,  # 使用 SMILES 作为 ID
                    "type": "chemical",
                    "smiles": smiles,
                    # 只有深度大于0的节点（前体）如果是可购买的，才标记为终端
                    # 目标分子（depth=0）不应该被标记为终端，即使它是可购买的
                    "terminal": terminal,
                    "plausibility": node.get("data", {}).get("policy_score", 0),
                    "depth": depth,
                    "precursor_score": node.get("data", {}).get("mcts_value", 0),
                    "asReactant": True,   # 使用驼峰命名（JavaScript 标准）
                    "asProduct": True,    # 使用驼峰命名（JavaScript 标准）
                    # 添加 ppg 属性以兼容 Askcos 的 isNodeTerminal 函数
                    # 如果节点是可购买的，设置 ppg=1（非零值表示可购买）
                    "ppg": 1 if is_purchasable else None,
                    # 保留原始 Syntheseus 数据
                    "syntheseus_id": node.get("id"),
                    "syntheseus_type": node.get("type"),
                    "has_solution": node.get("has_solution", False),
                }
                created_nodes[askcos_node_id] = askcos_node

            node_smiles_ids.append(askcos_node_id)

        # 将 Syntheseus ID 映射到所有 SMILES ID 列表
        node_id_map[node["id"]] = node_smiles_ids

    # 将 created_nodes 转换为列表
    nodes = list(created_nodes.values())

    # 调试输出：打印所有化学节点的 terminal 状态
    print(f"[DEBUG] Route {route_data.get('route_index', '?')} - Chemical nodes terminal status:")
    chemical_nodes = [n for n in nodes if n.get("type") == "chemical"]
    for node in chemical_nodes:
        print(f"  {node['smiles']}: terminal={node.get('terminal')}, depth={node.get('depth')}")
    print(f"[DEBUG] Total chemical nodes: {len(chemical_nodes)}, Total nodes (including reactions): {len(nodes)}")

    # 处理边和反应（将反应转换为独立节点）
    for edge in route_data["edges"]:
        reaction = edge.get("reaction", {})
        if not reaction:
            continue

        # 使用唯一 UUID 作为反应节点 ID
        rxn_node_id = str(uuid4())
        rxn_smiles = reaction.get("reaction_smiles", "")

        # 获取产物 SMILES（优先使用 reaction.product 字段）
        product_smiles = reaction.get("product", "")
        if not product_smiles and rxn_smiles and ">>" in rxn_smiles:
            # 如果没有 product 字段，从 reaction_smiles 解析
            product_smiles = rxn_smiles.split(">>")[1]

        if not product_smiles:
            continue  # 没有产物 SMILES，跳过此反应

        # 获取反应物列表
        reactants = reaction.get("reactants", [])
        if not reactants:
            # 如果没有 reactants 字段，尝试从 reaction_smiles 解析
            if rxn_smiles and ">>" in rxn_smiles:
                reactants = rxn_smiles.split(">>")[0].split(".")
            else:
                continue

        # 创建反应节点
        probability = reaction.get("metadata", {}).get("probability", 0)
        rxn_node = {
            "id": rxn_node_id,  # 使用 UUID 作为 ID
            "type": "reaction",
            "smiles": rxn_smiles,
            "plausibility": probability,
            "inVis": {},  # 必需属性：用于跟踪反应在显示图中的可见性
            # 添加 makeReactionDisplayNode 需要的属性（兼容 Askcos 格式）
            "model_metadata": [
                {
                    "direction": "retro",
                    "backend": "syntheseus",
                    "model_name": "syntheseus_mcts",
                    "model_score": probability,
                    "normalized_rank": 1,
                }
            ],
            "rxn_score_from_model": probability,
            "precursor_rank": 1,
            "precursor_score": probability,
            "precursor_properties": {},
            "rank": 1,
        }
        nodes.append(rxn_node)

        # 获取所有化学节点 ID（用于验证）
        chemical_ids = set(n.get("id") for n in nodes if n.get("type") == "chemical")

        # 创建边：产物 -> 反应
        if product_smiles in chemical_ids:
            edges.append({
                "id": str(uuid4()),
                "from": product_smiles,
                "to": rxn_node_id
            })

        # 创建边：反应 -> 反应物
        for reactant in reactants:
            if reactant in chemical_ids:
                edges.append({
                    "id": str(uuid4()),
                    "from": rxn_node_id,
                    "to": reactant
                })

    # 计算路径深度
    depths = [n["depth"] for n in nodes if n["type"] == "chemical"]
    depth = max(depths) if depths else 0

    # 计算反应节点相关的统计信息
    reaction_nodes = [n for n in nodes if n["type"] == "reaction"]
    num_reactions = len(reaction_nodes)

    # 计算可信度统计（从反应节点的 plausibility）
    plausibilities = [n.get("plausibility", 0) for n in reaction_nodes if n.get("plausibility", 0) > 0]
    avg_plausibility = sum(plausibilities) / len(plausibilities) if plausibilities else 0.0
    min_plausibility = min(plausibilities) if plausibilities else 0.0

    # 计算分数统计（从化学节点的 policy_score 或 precursor_score）
    chemical_nodes = [n for n in nodes if n["type"] == "chemical"]
    scores = []
    for n in chemical_nodes:
        # 优先使用 precursor_score (mcts_value)，其次使用 plausibility (policy_score)
        score = n.get("precursor_score", 0)
        if score == 0:
            score = n.get("plausibility", 0)
        if score > 0:
            scores.append(score)

    avg_score = sum(scores) / len(scores) if scores else 0.0
    min_score = min(scores) if scores else 0.0

    return {
        "id": f"route_{route_data.get('route_index', 0)}",
        "nodes": nodes,
        "edges": edges,
        "graph": {
            "depth": depth,
            "num_reactions": num_reactions,
            "avg_plausibility": round(avg_plausibility, 4),
            "min_plausibility": round(min_plausibility, 4),
            "avg_score": round(avg_score, 4),
            "min_score": round(min_score, 4),
            # 可选字段 - 使用默认值欺骗前端
            "precursor_cost": 0.0,
            "atom_economy": 0.0,
            "pmi": 0.0,
        },
        # 保留原始 Syntheseus 数据
        "syntheseus_data": {
            "route_index": route_data.get("route_index"),
            "graph_type": route_data.get("graph_type"),
            "num_nodes": route_data.get("num_nodes"),
            "num_edges": route_data.get("num_edges"),
        }
    }


def convert_syntheseus_graph_to_askcos(graph_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    将 Syntheseus 完整图数据转换为 Askcos 格式

    Args:
        graph_data: Syntheseus 图数据

    Returns:
        Askcos 兼容格式的图数据
    """
    paths = []

    # 如果 graph_data 包含多条路径
    if "paths" in graph_data:
        for path in graph_data["paths"]:
            paths.append(convert_to_askcos_format(path))
    else:
        # 单条路径
        paths.append(convert_to_askcos_format(graph_data))

    return {
        "paths": paths,
        "target_smiles": graph_data.get("root_smiles", ""),
        "stats": {
            "total_paths": len(paths),
            "total_chemicals": sum(len(p["nodes"]) for p in paths),
            "total_reactions": sum(len(p["edges"]) // 2 for p in paths) if paths else 0
        }
    }


def wrap_uds_to_askcos_response(
    uds_data: Dict[str, Any],
    task_id: str,
    target_smiles: str,
    stats_override: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    将 UDS 数据包装成 Askcos 完整响应格式

    返回与 askcos_return1.json 一致的 JSON 结构

    Args:
        uds_data: UDS 格式数据（包含 graph, node_dict, pathways 等）
        task_id: 任务 ID
        target_smiles: 目标分子 SMILES
        stats_override: 可选的统计信息覆盖

    Returns:
        完整的 Askcos 响应格式
    """
    from uuid import uuid4

    # 提取统计信息
    node_dict = uds_data.get("node_dict", {})
    chemical_nodes = [n for n in node_dict.values() if n.get("type") == "chemical"]
    reaction_nodes = [n for n in node_dict.values() if n.get("type") == "reaction"]

    # 如果提供了覆盖的统计信息，使用它；否则根据数据计算
    if stats_override:
        stats = stats_override.copy()
    else:
        stats = {
            "total_iterations": 1,
            "total_chemicals": len(chemical_nodes),
            "total_reactions": len(reaction_nodes),
            "total_templates": len(reaction_nodes),
            "total_paths": len(uds_data.get("pathways", [])),
            "first_path_time": 0.0,
            "build_time": 0.0,
            "path_time": 0.0
        }

    return {
        "task_id": task_id,
        "state": "SUCCESS",
        "complete": True,
        "failed": False,
        "percent": 1,
        "message": "Task complete!",
        "output": {
            "status_code": 200,
            "message": "mcts.call_raw() successfully executed.",
            "result": {
                "stats": stats,
                "uds": uds_data,
                "version": 2,
                "result_id": str(uuid4())
            }
        }
    }


def load_uds_from_results(results_dir: str) -> Dict[str, Any]:
    """
    从结果目录加载 UDS 格式数据

    Args:
        results_dir: 结果目录路径

    Returns:
        UDS 格式数据，如果文件不存在则返回 None
    """
    from pathlib import Path

    uds_path = Path(results_dir) / "uds_askcos.json"
    if not uds_path.exists():
        return None

    with open(uds_path, 'r', encoding='utf-8') as f:
        import json
        return json.load(f)
