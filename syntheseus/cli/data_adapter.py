"""
Syntheseus 到 Askcos 数据格式转换模块

将 Syntheseus 的路径数据格式转换为 Askcos-vue-nginx 前端可识别的格式。
"""
from typing import Dict, Any
from uuid import uuid4

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
