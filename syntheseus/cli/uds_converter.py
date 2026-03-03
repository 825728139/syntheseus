"""
UDS (Undirected Synthesis Graph) 格式转换模块

将 Syntheseus 的 output_graph 转换为与 Askcos 前端兼容的 UDS 格式。

UDS 格式包含：
- graph: 边列表 [{source: smiles, target: smiles}, ...] （使用 SMILES）
- node_dict: 节点字典 {smiles: node_info}
- pathways: 路径数组 [[{source: uuid, target: uuid}, ...], ...] （使用 UUID）
- pathways_properties: 路径属性数组
- uuid2smiles: UUID 到 SMILES 的映射字典

注意：graph 使用 SMILES 作为节点标识，pathways 使用 UUID 并通过 uuid2smiles 映射。
"""
from typing import Dict, List, Any, Iterator, Tuple
import json
from uuid import uuid4

# 导入图类型用于类型检查
try:
    from syntheseus.search.graph.and_or import AndOrGraph
    from syntheseus.search.graph.molset import MolSetGraph
except ImportError:
    AndOrGraph = None
    MolSetGraph = None


def convert_graph_to_uds(
    output_graph,
    routes_iterator: Iterator,
    stats: Dict[str, Any],
    max_routes: int = 50
) -> Dict[str, Any]:
    """
    将 Syntheseus 图转换为 Askcos UDS 格式

    Args:
        output_graph: Syntheseus 搜索结果图 (AndOrGraph 或 MolSetGraph)
        routes_iterator: 路径迭代器 (iter_routes_time_order 的结果)
        stats: 搜索统计信息
        max_routes: 最大路径数

    Returns:
        UDS 格式的字典，包含 graph, node_dict, pathways, pathways_properties, uuid2smiles
    """
    # 1. 构建 node_dict（化学节点和反应节点）
    node_dict = _build_node_dict(output_graph)

    # 2. 构建 graph 边数组（使用 SMILES）
    graph_edges = _build_graph_edges(output_graph)

    # 3. 提取所有路径
    routes = list(routes_iterator)

    # 4. 构建 pathways 数组（使用 UUID）和 uuid2smiles 映射
    pathways, pathways_properties, uuid2smiles = _build_pathways(output_graph, routes)

    return {
        "node_dict": node_dict,
        "graph": graph_edges,
        "pathways": pathways,
        "pathways_properties": pathways_properties,
        "uuid2smiles": uuid2smiles
    }


def _build_node_dict(output_graph) -> Dict[str, Dict[str, Any]]:
    """
    构建 node_dict，包含化学节点和反应节点

    Returns:
        节点字典 {smiles: node_info}，其中化学节点使用分子 SMILES 作为 key，
        反应节点使用反应 SMILES 作为 key
    """
    from json_graph import serialize_node, serialize_reaction

    node_dict = {}
    chemical_counts = {}  # 统计化学节点作为反应物/产物的次数

    # 首先处理所有化学节点
    for node in output_graph.nodes():
        serialized = serialize_node(node)

        # 提取化学节点信息
        node_smiles = None
        is_purchasable = False
        depth = 0

        if 'smiles' in serialized:
            node_smiles = serialized['smiles']
            is_purchasable = serialized.get('is_purchasable', False)
            depth = serialized.get('depth', 0)
        elif 'smiles_list' in serialized and serialized['smiles_list']:
            node_smiles = serialized['smiles_list'][0]
            is_purchasable = serialized.get('is_purchasable', False)
            depth = serialized.get('depth', 0)

        if node_smiles:
            # 初始化计数
            if node_smiles not in chemical_counts:
                chemical_counts[node_smiles] = {"as_reactant": 0, "as_product": 0}

            # 如果节点还不存在，创建化学节点
            if node_smiles not in node_dict:
                # terminal 属性：只有深度 > 0 且可购买的节点才标记为 terminal
                terminal = is_purchasable and depth > 0

                node_dict[node_smiles] = {
                    "smiles": node_smiles,
                    "as_reactant": 0,  # 稍后更新
                    "as_product": 0,   # 稍后更新
                    "properties": [
                        {"link": ""},
                        {"availability": ""}
                    ],
                    "purchase_price": 1.0 if is_purchasable else 0.0,
                    "terminal": terminal,
                    "type": "chemical",
                    "id": node_smiles
                }

    # 处理反应节点（从边中提取）
    for source, target in output_graph._graph.edges:
        edge_data = output_graph._graph.get_edge_data(source, target)

        if 'reaction' in edge_data:
            reaction = edge_data['reaction']
            rxn_serialized = serialize_reaction(reaction)
            reaction_smiles = rxn_serialized['reaction_smiles']
            product = rxn_serialized.get('product', '')
            reactants = rxn_serialized.get('reactants', [])

            # === 确保 all 反应物和产物都有化学节点 ===
            # 收集所有相关的化学分子
            all_chemicals = []
            if product:
                all_chemicals.append(product)
            all_chemicals.extend(reactants)

            for chem_smiles in all_chemicals:
                if chem_smiles and chem_smiles not in node_dict:
                    # 创建缺失的化学节点（使用默认值）
                    node_dict[chem_smiles] = {
                        "smiles": chem_smiles,
                        "as_reactant": 0,
                        "as_product": 0,
                        "properties": [
                            {"link": ""},
                            {"availability": ""}
                        ],
                        "purchase_price": 0.0,  # 未知价格
                        "terminal": False,     # 未知是否可购买
                        "type": "chemical",
                        "id": chem_smiles
                    }
                    # 初始化计数（如果尚未初始化）
                    if chem_smiles not in chemical_counts:
                        chemical_counts[chem_smiles] = {"as_reactant": 0, "as_product": 0}

            # 更新化学节点计数
            if product in chemical_counts:
                chemical_counts[product]['as_product'] += 1
            for reactant in reactants:
                if reactant in chemical_counts:
                    chemical_counts[reactant]['as_reactant'] += 1

            # 创建反应节点（使用反应 SMILES 作为 id）
            if reaction_smiles and reaction_smiles not in node_dict:
                # 从 metadata 中提取概率等属性
                metadata = rxn_serialized.get('metadata', {})
                probability = metadata.get('probability', 0.0)

                # 构建反应节点
                reaction_node = {
                    "smiles": reaction_smiles,
                    "plausibility": probability,
                    "rxn_score_from_model": probability,
                    "model_metadata": [
                        {
                            "direction": "retro",
                            "backend": "template_relevance",
                            "model_name": "syntheseus",
                            "attributes": {
                                "max_num_templates": 1000,
                                "max_cum_prob": 0.999,
                                "attribute_filter": []
                            },
                            "model_score": probability,
                            "normalized_model_score": probability,
                            "rank": 1,
                            "reaction_id": None,
                            "reaction_set": None,
                            "source": {
                                "template": {
                                    "count": 1,
                                    "dimer_only": False,
                                    "index": 0,
                                    "intra_only": False,
                                    "necessary_reagent": "",
                                    "reaction_smarts": reaction_smiles,
                                    "template_set": "syntheseus",
                                    "_id": reaction_smiles,
                                    "template_score": probability,
                                    "template_rank": 1,
                                    "num_examples": 1
                                }
                            }
                        }
                    ],
                    "precursor_properties": {
                        "rms_molwt": 0.0,
                        "num_rings": 0,
                        "scscore": 0.0
                    },
                    "precursor_rank": 1,
                    "precursor_score": probability,
                    "reaction_properties": {
                        "canonical_reaction_smiles": reaction_smiles,
                        "mapped_smiles": reaction_smiles,
                        "plausibility": probability,
                        "reacting_atoms": [],
                        "selec_error": None,
                        "cluster_id": None,
                        "cluster_name": None
                    },
                    "type": "reaction",
                    "id": reaction_smiles  # 使用反应 SMILES 作为 id
                }
                node_dict[reaction_smiles] = reaction_node

    # 更新化学节点的 as_reactant 和 as_product 计数
    for smiles, counts in chemical_counts.items():
        if smiles in node_dict:
            node_dict[smiles]["as_reactant"] = counts['as_reactant']
            node_dict[smiles]["as_product"] = counts['as_product']

    return node_dict


def _build_graph_edges(output_graph) -> List[Dict[str, str]]:
    """
    构建 graph 边数组（使用 SMILES 作为 source/target）

    Returns:
        边列表 [{source: smiles, target: smiles}, ...]
    """
    from json_graph import serialize_reaction

    edges = []
    seen_edges = set()  # 用于去重

    for source, target in output_graph._graph.edges:
        edge_data = output_graph._graph.get_edge_data(source, target)

        if 'reaction' in edge_data:
            reaction = edge_data['reaction']
            rxn_serialized = serialize_reaction(reaction)
            reaction_smiles = rxn_serialized['reaction_smiles']
            product = rxn_serialized.get('product', '')
            reactants = rxn_serialized.get('reactants', [])

            # 产物 -> 反应（使用 SMILES）
            if product:
                edge_key = (product, reaction_smiles)
                if edge_key not in seen_edges:
                    edges.append({"source": product, "target": reaction_smiles})
                    seen_edges.add(edge_key)

            # 反应 -> 反应物（使用 SMILES）
            for reactant in reactants:
                edge_key = (reaction_smiles, reactant)
                if edge_key not in seen_edges:
                    edges.append({"source": reaction_smiles, "target": reactant})
                    seen_edges.add(edge_key)

    return edges


def _build_pathways(
    output_graph,
    routes: List
) -> Tuple[List[List[Dict[str, str]]], List[Dict[str, Any]], Dict[str, str]]:
    """
    构建 pathways 数组和 pathways_properties 数组（pathways 使用 UUID）

    Returns:
        (pathways, pathways_properties, uuid2smiles) 元组
    """
    from json_graph import serialize_node, serialize_reaction

    pathways = []
    pathways_properties = []
    uuid2smiles = {}  # 最终输出: {uuid: smiles}
    global_smiles_to_uuid = {}  # 全局查找: {smiles: uuid}

    # 为根节点（目标分子）分配固定 UUID
    ROOT_UUID = "00000000-0000-0000-0000-000000000000"

    for route_idx, route in enumerate(routes):
        route_list = list(route)
        pathway_edges = []
        max_depth = 0
        total_cost = 0.0

        # SMILES -> UUID 映射（当前路径，合并全局映射）
        smiles_to_uuid = {}

        # 首先计算路径的最大深度
        for node in route_list:
            serialized = serialize_node(node)
            depth = serialized.get('depth', 0)
            max_depth = max(max_depth, depth)

        # 提取路径中的边
        # 使用类型检查（兼容导入失败的情况）
        is_molset = (MolSetGraph is not None and isinstance(output_graph, MolSetGraph)) or \
                    type(output_graph).__name__ == 'MolSetGraph'
        is_andor = (AndOrGraph is not None and isinstance(output_graph, AndOrGraph)) or \
                   type(output_graph).__name__ == 'AndOrGraph'

        if is_molset:
            # MolSetGraph: 路径是有序节点序列，提取相邻节点间的边
            for i in range(len(route_list) - 1):
                u, v = route_list[i], route_list[i + 1]
                edge_data = output_graph._graph.get_edge_data(u, v)

                if edge_data and "reaction" in edge_data:
                    reaction = edge_data["reaction"]
                    rxn_serialized = serialize_reaction(reaction)
                    reaction_smiles = rxn_serialized['reaction_smiles']
                    product = rxn_serialized.get('product', '')
                    reactants = rxn_serialized.get('reactants', [])

                    # 为产物生成/分配 UUID
                    if product and product not in smiles_to_uuid:
                        # 检查是否为根节点（depth=0）
                        serialized_u = serialize_node(u)
                        if serialized_u.get('depth', 0) == 0:
                            smiles_to_uuid[product] = ROOT_UUID
                        elif product in global_smiles_to_uuid:
                            # 已在全局映射中，复用 UUID
                            smiles_to_uuid[product] = global_smiles_to_uuid[product]
                        else:
                            smiles_to_uuid[product] = str(uuid4())

                    # 为反应生成 UUID
                    if reaction_smiles and reaction_smiles not in smiles_to_uuid:
                        if reaction_smiles in global_smiles_to_uuid:
                            smiles_to_uuid[reaction_smiles] = global_smiles_to_uuid[reaction_smiles]
                        else:
                            smiles_to_uuid[reaction_smiles] = str(uuid4())

                    # 为反应物生成 UUIDs
                    for reactant in reactants:
                        if reactant and reactant not in smiles_to_uuid:
                            if reactant in global_smiles_to_uuid:
                                smiles_to_uuid[reactant] = global_smiles_to_uuid[reactant]
                            else:
                                smiles_to_uuid[reactant] = str(uuid4())

                    # 产物 -> 反应（使用 UUID）
                    if product and reaction_smiles:
                        pathway_edges.append({
                            "source": smiles_to_uuid[product],
                            "target": smiles_to_uuid[reaction_smiles]
                        })

                    # 反应 -> 反应物（使用 UUID）
                    for reactant in reactants:
                        if reactant:
                            pathway_edges.append({
                                "source": smiles_to_uuid[reaction_smiles],
                                "target": smiles_to_uuid[reactant]
                            })

                    # 累加成本（基于概率）
                    probability = rxn_serialized.get('metadata', {}).get('probability', 0.0)
                    if probability > 0:
                        total_cost += -probability  # 负概率作为成本

        elif is_andor:
            # AndOrGraph: 处理 OrNode 和 AndNode
            for i in range(len(route_list) - 1):
                u, v = route_list[i], route_list[i + 1]
                u_serialized = serialize_node(u)

                # 如果是 AndNode（反应节点）
                if u_serialized.get('type') == 'AndNode':
                    reaction_smiles = u_serialized.get('reaction_smiles', '')
                    product = u_serialized.get('product', '')
                    reactants = u_serialized.get('reactants', [])

                    # 为产物生成/分配 UUID
                    if product and product not in smiles_to_uuid:
                        if product in uuid2smiles:
                            smiles_to_uuid[product] = uuid2smiles[product]
                        else:
                            smiles_to_uuid[product] = str(uuid4())

                    # 为反应生成 UUID
                    if reaction_smiles and reaction_smiles not in smiles_to_uuid:
                        if reaction_smiles in uuid2smiles:
                            smiles_to_uuid[reaction_smiles] = uuid2smiles[reaction_smiles]
                        else:
                            smiles_to_uuid[reaction_smiles] = str(uuid4())

                    # 为反应物生成 UUIDs
                    for reactant in reactants:
                        if reactant and reactant not in smiles_to_uuid:
                            if reactant in uuid2smiles:
                                smiles_to_uuid[reactant] = uuid2smiles[reactant]
                            else:
                                smiles_to_uuid[reactant] = str(uuid4())

                    # 产物 -> 反应（使用 UUID）
                    if product and reaction_smiles:
                        pathway_edges.append({
                            "source": smiles_to_uuid[product],
                            "target": smiles_to_uuid[reaction_smiles]
                        })

                    # 反应 -> 反应物（使用 UUID）
                    for reactant in reactants:
                        if reactant:
                            pathway_edges.append({
                                "source": smiles_to_uuid[reaction_smiles],
                                "target": smiles_to_uuid[reactant]
                            })

        # 更新全局映射
        for smiles, uuid in smiles_to_uuid.items():
            # 更新 smiles->uuid 映射（用于跨路径查找）
            if smiles not in global_smiles_to_uuid:
                global_smiles_to_uuid[smiles] = uuid
            # 更新 uuid->smiles 映射（最终输出格式）
            if uuid not in uuid2smiles:
                uuid2smiles[uuid] = smiles

        if pathway_edges:
            pathways.append(pathway_edges)
            pathways_properties.append({
                "depth": max_depth,  # 使用计算得到的最大深度
                "precursor_cost": abs(total_cost) if total_cost != 0 else 1.0,
                "score": None,
                "cluster_id": None
            })

    return pathways, pathways_properties, uuid2smiles
