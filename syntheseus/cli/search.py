"""Script for running end-to-end retrosynthetic search.

The supported single-step model types are listed in `syntheseus/cli/eval_single_step.py`;
each can be combined with Retro*, MCTS, or PDVN to perform search.

Example invocation:
    python ./syntheseus/cli/search.py \
        inventory_smiles_file=[INVENTORY_SMILES_FILE_PATH] \
        search_target="NC1=Nc2ccc(F)cc2C2CCCC12" \
        model_class=LocalRetro \
        model_dir=[MODEL_DIR] \
        time_limit_s=60
    python ./syntheseus/cli/search.py \
        inventory_smiles_file=emolecules.txt \
        search_target="NC1=Nc2ccc(F)cc2C2CCCC12" \
        model_class=RetroKNN \
        model_dir=/home/liwenlong/.cache/torch/syntheseus/RetroKNN_backward \
        time_limit_s=1800 \
        search_algorithm=mcts \
        results_dir=mcts_results/ \
        use_gpu=False \
        num_routes_to_plot=10
"""

from __future__ import annotations

import datetime
import json
import logging
import math
import pickle
import statistics
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, Iterator, List, Optional, cast
from uuid import uuid4

import yaml
from omegaconf import MISSING, DictConfig, OmegaConf
from tqdm import tqdm

from syntheseus import Molecule
from syntheseus.reaction_prediction.inference.config import BackwardModelConfig
from syntheseus.reaction_prediction.utils.config import get_config as cli_get_config
from syntheseus.reaction_prediction.utils.misc import set_random_seed
from syntheseus.reaction_prediction.utils.model_loading import get_model
from syntheseus.search import INT_INF
from syntheseus.search.algorithms.best_first.retro_star import RetroStarSearch
from syntheseus.search.algorithms.mcts import base as mcts_base
from syntheseus.search.algorithms.mcts.molset import MolSetMCTS
from syntheseus.search.algorithms.pdvn import PDVN_MCTS
from syntheseus.search.analysis.route_extraction import iter_routes_time_order, iter_routes_cost_order
from syntheseus.search.analysis.solution_time import get_first_solution_time
from syntheseus.search.graph.and_or import AndOrGraph
from syntheseus.search.graph.molset import MolSetGraph
from syntheseus.search.mol_inventory import SmilesListInventory
from syntheseus.search.node_evaluation import common as node_evaluation_common
from syntheseus.search.utils.misc import lookup_by_name

try:
    # Try to import the visualization code, which will work only if `graphviz` is installed.
    from syntheseus.search.visualization import visualize_andor, visualize_molset

    VISUALIZATION_CODE_IMPORTED = True
except ModuleNotFoundError:
    VISUALIZATION_CODE_IMPORTED = False

logger = logging.getLogger(__file__)


# ============================================================================
# Interactive Incremental Search Support
# ============================================================================


def print_interim_stats(
    graph: AndOrGraph | MolSetGraph,
    iteration: int,
    total_time: float,
    rxn_model,
    target: Molecule,
    mol_inventory: SmilesListInventory,
) -> None:
    """打印中间搜索结果统计（完整输出）.

    Args:
        graph: 搜索图
        iteration: 当前迭代次数（从0开始）
        total_time: 总搜索时间（秒）
        rxn_model: 反应模型
        target: 目标分子
        mol_inventory: 分子库存
    """
    # 1. 基本统计信息
    num_nodes = len(graph)
    rxn_calls = rxn_model.num_calls()

    # 2. 检查是否有解
    has_solution = graph.root_node.has_solution

    # 3. 统计解的数量（通过遍历路径）
    num_solutions = 0
    try:
        for _ in iter_routes_time_order(graph, max_routes=1000):
            num_solutions += 1
    except Exception:
        # 如果无法统计，使用估计值
        num_solutions = -1 if has_solution else 0

    print(f"迭代轮次: {iteration + 1}")
    print(f"累计搜索时间: {total_time:.1f} 秒")
    print(f"图中节点总数: {num_nodes}")
    print(f"反应模型调用次数: {rxn_calls}")
    print(f"目标分子: {target.smiles}")
    print(f"目标在库存中: {mol_inventory.is_purchasable(target)}")
    print(f"是否找到解: {'是' if has_solution else '否'}")

    if has_solution and num_solutions > 0:
        print(f"已发现解的数量: {num_solutions}")


@dataclass
class RetroStarConfig:
    max_expansion_depth: int = 10

    value_function_class: str = "ConstantNodeEvaluator"
    value_function_kwargs: Dict[str, Any] = field(default_factory=lambda: {"constant": 0.0})

    and_node_cost_fn_class: str = "ReactionModelLogProbCost"
    and_node_cost_fn_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MCTSConfig:
    max_expansion_depth: int = 20

    value_function_class: str = "ConstantNodeEvaluator"
    value_function_kwargs: Dict[str, Any] = field(default_factory=lambda: {"constant": 0.5})

    reward_function_class: str = "HasSolutionValueFunction"
    reward_function_kwargs: Dict[str, Any] = field(default_factory=dict)

    policy_class: str = "ReactionModelProbPolicy"
    policy_kwargs: Dict[str, Any] = field(default_factory=dict)

    bound_constant: float = 1.0
    bound_function_class: str = "pucb_bound"


@dataclass
class PDVNConfig:
    max_expansion_depth: int = 10

    value_function_syn_class: str = "ConstantNodeEvaluator"
    value_function_syn_kwargs: Dict[str, Any] = field(default_factory=lambda: {"constant": 0.5})

    value_function_cost_class: str = "ConstantNodeEvaluator"
    value_function_cost_kwargs: Dict[str, Any] = field(default_factory=lambda: {"constant": 0.0})

    and_node_cost_fn_class: str = "ConstantNodeEvaluator"
    and_node_cost_fn_kwargs: Dict[str, Any] = field(default_factory=lambda: {"constant": 0.1})

    policy_class: str = "ReactionModelProbPolicy"
    policy_kwargs: Dict[str, Any] = field(default_factory=dict)

    c_dead: float = 5.0
    bound_constant: float = 1e2
    bound_function_class: str = "pucb_bound"


class SearchAlgorithmClass(Enum):
    retro_star = RetroStarSearch
    mcts = MolSetMCTS
    pdvn = PDVN_MCTS


@dataclass
class SearchAlgorithmConfig:
    search_algorithm: SearchAlgorithmClass = SearchAlgorithmClass.retro_star
    retro_star_config: RetroStarConfig = field(default_factory=RetroStarConfig)
    mcts_config: MCTSConfig = field(default_factory=MCTSConfig)
    pdvn_config: PDVNConfig = field(default_factory=PDVNConfig)

    # By default limit search time (but set very high iteration limits just in case)
    time_limit_s: float = 600
    limit_reaction_model_calls: int = 1_000_000
    limit_iterations: int = 1_000_000
    limit_graph_nodes: int = INT_INF
    prevent_repeat_mol_in_trees: bool = True
    stop_on_first_solution: bool = False
    expand_purchasable_target: bool = False  # Whether to expand target even if it's purchasable


def search_algorithm_config_to_kwargs(config: SearchAlgorithmConfig) -> Dict[str, Any]:
    alg_kwargs = {
        key: cast(DictConfig, config).get(key)
        for key in [
            "time_limit_s",
            "limit_reaction_model_calls",
            "limit_iterations",
            "limit_graph_nodes",
            "prevent_repeat_mol_in_trees",
            "stop_on_first_solution",
            "expand_purchasable_target",
        ]
    }

    def build_node_evaluator(key: str) -> None:
        # Build a node evaluator based on chosen class and args
        alg_kwargs[key] = lookup_by_name(node_evaluation_common, alg_kwargs[f"{key}_class"])(
            **alg_kwargs[f"{key}_kwargs"]
        )

        # Delete the arguments to avoid passing them into the algorithm's constructor downstream
        del alg_kwargs[f"{key}_class"]
        del alg_kwargs[f"{key}_kwargs"]

    if config.search_algorithm == SearchAlgorithmClass.retro_star:
        alg_kwargs.update(cast(Dict[str, Any], OmegaConf.to_container(config.retro_star_config)))
        build_node_evaluator("value_function")
        build_node_evaluator("and_node_cost_fn")
    elif config.search_algorithm == SearchAlgorithmClass.mcts:
        alg_kwargs.update(cast(Dict[str, Any], OmegaConf.to_container(config.mcts_config)))
        build_node_evaluator("value_function")
        build_node_evaluator("reward_function")
        build_node_evaluator("policy")

        alg_kwargs["bound_function"] = lookup_by_name(mcts_base, alg_kwargs["bound_function_class"])
        del alg_kwargs["bound_function_class"]
    elif config.search_algorithm == SearchAlgorithmClass.pdvn:
        alg_kwargs.update(cast(Dict[str, Any], OmegaConf.to_container(config.pdvn_config)))
        build_node_evaluator("value_function_syn")
        build_node_evaluator("value_function_cost")
        build_node_evaluator("and_node_cost_fn")
        build_node_evaluator("policy")

        alg_kwargs["bound_function"] = lookup_by_name(mcts_base, alg_kwargs["bound_function_class"])
        del alg_kwargs["bound_function_class"]

    return alg_kwargs


@dataclass
class BaseSearchConfig(SearchAlgorithmConfig):
    # Molecule(s) to search for (either as a single explicit SMILES or a file)
    search_target: str = MISSING
    search_targets_file: str = MISSING

    inventory_smiles_file: str = MISSING  # Purchasable molecules
    results_dir: str = "."  # Directory to save the results in
    append_timestamp_to_dir: bool = True  # Whether to append the current time to directory name

    use_gpu: bool = True  # Whether to use a GPU
    canonicalize_inventory: bool = False  # Whether to canonicalize the inventory SMILES

    # Fields configuring the reaction model (on top of the arguments from `BackwardModelConfig`)
    num_top_results: int = 50  # Number of results to request
    reaction_model_use_cache: bool = True  # Whether to cache the results

    # Fields configuring what to save after the run
    save_graph: bool = True  # Whether to save the full reaction graph (can be large)
    num_routes_to_plot: int = 5  # Number of routes to extract and plot for a quick check

    # Fields for resume search mode
    resume_search: Optional[str] = None  # Path to existing search results folder to resume from
    resume_search_time_s: float = 30.0  # Time limit for resumed search (default 30 seconds)

@dataclass
class SearchConfig(BackwardModelConfig, BaseSearchConfig):
    """Config for running search for given search targets."""

    pass


def merge_uds_data(old_uds: dict, new_uds: dict) -> dict:
    """合并新旧 UDS 数据（直接合并，不排序）

    Args:
        old_uds: 旧的 UDS 数据（包含 node_dict, graph, pathways, pathways_properties, uuid2smiles）
        new_uds: 新的 UDS 数据（相同结构）

    Returns:
        合并后的 UDS 数据
    """
    if not old_uds:
        return new_uds
    if not new_uds:
        return old_uds

    # 合并 node_dict（新节点信息）
    merged_node_dict = {**old_uds.get('node_dict', {}), **new_uds.get('node_dict', {})}

    # 合并 graph（路径到节点的映射关系）
    merged_graph = old_uds.get('graph', []) + new_uds.get('graph', [])

    # 合并 uuid2smiles
    merged_uuid2smiles = {**old_uds.get('uuid2smiles', {}), **new_uds.get('uuid2smiles', {})}

    # 合并 pathways 和 pathways_properties
    merged_uds = {
        'node_dict': merged_node_dict,
        'graph': merged_graph,
        'pathways': old_uds.get('pathways', []) + new_uds.get('pathways', []),
        'pathways_properties': old_uds.get('pathways_properties', []) + new_uds.get('pathways_properties', []),
        'uuid2smiles': merged_uuid2smiles,
    }

    return merged_uds


def run_from_config(config: SearchConfig) -> Path:
    set_random_seed(0)

    # 配置日志: 同时输出到终端和文件
    log_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 终端处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(log_formatter)

    # 为根logger添加处理器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)

    # 文件处理器将在 results_dir 创建后添加

    print("Running search with the following config:")
    print(config)

    if config.num_routes_to_plot > 0 and not VISUALIZATION_CODE_IMPORTED:
        raise ValueError(
            "Could not import visualization code (likely `viz` dependencies are not installed); "
            "please install missing dependencies or set `num_routes_to_plot=0`"
        )

    search_target, search_targets_file = [
        cast(DictConfig, config).get(key) for key in ["search_target", "search_targets_file"]
    ]

    if not ((search_target is None) ^ (search_targets_file is None)):
        raise ValueError(
            "Exactly one of 'search_target' and 'search_targets_file' should be provided"
        )

    # Prepare the search targets
    search_targets: List[str] = []
    if search_target is not None:
        search_targets = [search_target]
    else:
        with open(config.search_targets_file, "rt") as f_targets:
            search_targets = [line.strip() for line in f_targets]

    if not config.save_graph and config.num_routes_to_plot == 0:
        logger.warning(
            "Neither 'save_graph' nor 'num_routes_to_plot' is set; output saved will be minimal"
        )

    # Load the single-step model
    search_rxn_model = get_model(  # type: ignore
        config,
        batch_size=1,
        num_gpus=int(config.use_gpu),
        use_cache=config.reaction_model_use_cache,
        default_num_results=config.num_top_results,
    )

    # Set up the inventory
    mol_inventory = SmilesListInventory.load_from_file(
        config.inventory_smiles_file, canonicalize=config.canonicalize_inventory
    )

    alg = config.search_algorithm.value(
        reaction_model=search_rxn_model,
        mol_inventory=mol_inventory,
        **search_algorithm_config_to_kwargs(config),
    )

    # 初始化恢复模式标志变量
    is_resume_mode = False
    # 检查是否存在之前的搜索结果以恢复
    if config.resume_search is None or not (Path(config.resume_search) / "graph.pkl").exists():
        # 运行旧有的搜索逻辑（从头开始）
        logger.info("No existing search graph found, starting a new search")
        # Prepare the output directory
        results_dir_top_level = Path(config.results_dir)

        dirname = config.model_class.name
        if config.append_timestamp_to_dir:
            timestamp = datetime.datetime.now().isoformat(timespec="seconds")
            dirname += f"_{str(timestamp)}"

        results_dir_current_run = results_dir_top_level / dirname

        # 添加文件日志处理器
        results_dir_current_run.mkdir(parents=True, exist_ok=True)
        log_file = results_dir_current_run / "search.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)

        logger.info(f"日志文件将保存到: {log_file}")
        logger.info("Setup completed")
        num_targets = len(search_targets)

        all_stats: List[Dict[str, Any]] = []
        for idx, smiles in enumerate(tqdm(search_targets)):
            logger.info(f"Running search for target {smiles}")

            if num_targets == 1:
                results_dir = results_dir_current_run
            else:
                results_dir = results_dir_current_run / str(idx)

            # results_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Outputs will be saved under {results_dir}")

            results_lock_path = results_dir / ".lock"
            results_stats_path = results_dir / "stats.json"

            if results_lock_path.exists():
                paths = [path for path in results_dir.iterdir() if path.is_file()]
                logger.warning(
                    f"Lockfile was found which means the last run failed, purging {len(paths)} files"
                )

                for path in paths:
                    path.unlink()
            elif results_stats_path.exists():
                with open(results_stats_path, "rt") as f_stats:
                    stats = json.load(f_stats)
                    if stats.get("index") != idx or stats.get("smiles") != smiles:
                        raise RuntimeError(
                            f"Data present under {results_dir} does not match the current run"
                        )

                    all_stats.append(stats)

                logger.info("Search results already exist, skipping")
                continue

            results_lock_path.touch()

            target = Molecule(smiles)
            target_in_inventory = mol_inventory.is_purchasable(target)

            if target_in_inventory:
                if config.expand_purchasable_target:
                    logger.info("Target is purchasable but will be expanded anyway")
                else:
                    logger.info(
                        "Search will be a no-op as the target is purchasable; "
                        "set `expand_purchasable_target` if you want to expand it regardless"
                    )

            # 保持原有行为
            alg.reset()
            output_graph, _ = alg.run_from_mol(target)
            logger.info(f"Finished search for target {smiles}")

        if config.save_graph:
            from json_graph import serialize_node, serialize_reaction
            node_to_id = {node: i for i, node in enumerate(output_graph._graph.nodes)}

            # 获取根节点的SMILES，兼容两种图类型：
            # 1. AndOrGraph (retro_star): 根节点是OrNode，有mol.smiles属性
            # 2. MolSetGraph (MCTS): 根节点是MolSetNode，有mols属性（frozenset[Molecule]）
            root_node = output_graph._root_node
            if hasattr(root_node, 'mol'):
                # AndOrGraph (retro_star)
                root_smiles = root_node.mol.smiles
            elif hasattr(root_node, 'mols'):
                # MolSetGraph (MCTS)
                root_smiles = list(root_node.mols)[0].smiles if len(root_node.mols) == 1 else None
                if root_smiles is None:
                    root_smiles = ','.join(sorted([mol.smiles for mol in root_node.mols]))
            else:
                root_smiles = None

            output = {
                'nodes': [serialize_node(n) for n in output_graph._graph.nodes],
                'edges': [
                    {
                        'source': node_to_id[s],
                        'target': node_to_id[t],
                        'reaction': serialize_reaction(output_graph._graph.edges[s, t]['reaction'])
                        if 'reaction' in output_graph._graph.edges[s, t] else None
                    }
                    for s, t in output_graph._graph.edges
                ],
                'root_node_id': node_to_id[root_node],
                'root_smiles': root_smiles,
            }

            with open(results_dir / "graph_output.json", 'w') as f:
                json.dump(output, f, indent=2)
            with open(results_dir / "graph.pkl", "wb") as f_graph:
                pickle.dump(output_graph, f_graph)

    else:
        # 集成恢复搜索功能
        results_dir = Path(config.resume_search)
        graph_pkl = results_dir / "graph.pkl"
        logger.info(f"Resume mode: loading graph from {graph_pkl}")

        # 加载之前的搜索图和输出结果
        with open(graph_pkl, "rb") as f:
            output_graph = pickle.load(f)
        uds_json = results_dir / "uds_askcos.json"
        if uds_json.exists():
            with open(uds_json, "r") as f:
                resume_old_uds = json.load(f)
            logger.info(f"Loaded {len(resume_old_uds.get('pathways', []))} existing pathways")
        else:
            resume_old_uds = None
            logger.info("No existing UDS data found, starting with empty UDS")
        alg.run_from_graph(output_graph)

        # 匹配兼容变量名，准备后续处理
        results_dir_current_run = results_dir
        results_lock_path = results_dir / ".lock"
        results_lock_path.touch()
        results_stats_path = results_dir / "stats.json"
        all_stats: List[Dict[str, Any]] = []

        # 恢复搜索：设置标志变量，供后续共享代码使用
        is_resume_mode = True

    # 初始化根节点信息（供后续统计和 UDS 构建使用）
    root_node = list(routes[0])[0]
    if hasattr(root_node, 'mol'):
        root_molecule = root_node.mol
    elif hasattr(root_node, 'mols'):
        root_molecule = list(root_node.mols)[0]
    
    # Time of first solution (rxn model calls)
    for node in output_graph.nodes():
        node.data["analysis_time"] = node.data["num_calls_rxn_model"]
    soln_time_rxn_model_calls = get_first_solution_time(output_graph)

    # Time of first solution (wallclock)
    for node in output_graph.nodes():
        node.data["analysis_time"] = (
            node.creation_time - output_graph.root_node.creation_time
        ).total_seconds()
    soln_time_wallclock = get_first_solution_time(output_graph)

    stats = {
        "index": "resume" if is_resume_mode else idx,
        "smiles": root_molecule.smiles,
        "target_in_inventory": "resume" if is_resume_mode else target_in_inventory,
        "rxn_model_calls_used": alg.reaction_model.num_calls(),
        "num_nodes_in_final_tree": len(output_graph),
        "soln_time_rxn_model_calls": soln_time_rxn_model_calls,
        "soln_time_wallclock": soln_time_wallclock,
    }

    all_stats.append(stats)
    logger.info(pformat(stats))
    if results_stats_path.exists():
        results_stats_path = results_stats_path.with_suffix(".jsonl")
        logger.info(f"Stats file already exists, appending to {results_stats_path}")
    with open(results_stats_path, "wt") as f_stats:
        f_stats.write(json.dumps(stats, indent=2))

    if config.num_routes_to_plot > 0:
        # Extract some synthesis routes in the order they were found
        logger.info(f"Extracting up to {config.num_routes_to_plot} routes for analysis")

        # TODO(kmaziarz): Add options to extract a diverse (or otherwise interesting) subset.
        # routes: Iterator = iter_routes_time_order(
        #     output_graph, max_routes=config.num_routes_to_plot
        # )

        # 恢复模式：只提取包含新节点的路径
        skip_uds_processing = False  # 标志：是否跳过 UDS 处理
        if is_resume_mode:
            routes: Iterator = iter_routes_cost_order(
                output_graph, max_routes=config.num_routes_to_plot, only_new_nodes=True
            )
            routes = list(routes)  # 转换为列表以检查数量
            logger.info(f"Resume mode: found {len(routes)} new routes")

            # 如果没有新路径，只保存 graph.pkl，跳过后续 UDS 处理
            if len(routes) == 0:
                logger.info("No new routes found, only saving updated graph.pkl")
                with open(results_dir / "graph.pkl", "wb") as f_graph:
                    pickle.dump(output_graph, f_graph)
                skip_uds_processing = True
        else:
            routes: Iterator = iter_routes_cost_order(
                output_graph, max_routes=config.num_routes_to_plot
            )
            routes = list(routes)

        # 如果跳过 UDS 处理，直接清理并进入下一个目标
        if skip_uds_processing:
            results_lock_path.unlink()
            del results_dir
        else:
            # 正常处理 UDS（原有代码）
            # 初始化 UDS 格式字典
            uds = {
                "node_dict": {},      # {smiles: node_info}
                "graph": [],          # [{source: smiles, target: smiles}, ...]
                "pathways": [],       # [[{source: uuid, target: uuid}, ...], ...]
                "pathways_properties": [],  # [{depth, precursor_cost, score, cluster_id}, ...]
                "uuid2smiles": {}     # {uuid: smiles}
            }
            global_smiles_to_uuid = {}  # 全局 SMILES 到 UUID 的映射
            ROOT_UUID = "00000000-0000-0000-0000-000000000000"
        
            uds["node_dict"][root_molecule.smiles] = {
                "smiles": root_molecule.smiles,
                "as_reactant": 1,
                "as_product": 1,
                "properties": [{"link": ""}, {"availability": ""}],
                "purchase_price": 1.0 if root_molecule.metadata['is_purchasable'] else False,
                "terminal": False,
                "type": "chemical",
                "id": root_molecule.smiles
            }
            # 为根节点分配固定 UUID
            global_smiles_to_uuid[root_molecule.smiles] = ROOT_UUID
            uds["uuid2smiles"][ROOT_UUID] = root_molecule.smiles

            resume_routes_num = len(resume_old_uds.get('pathways', [])) if resume_old_uds else 0
            for route_idx, route in enumerate(routes):
                route_idx += resume_routes_num  # 如果是恢复模式，继续之前的索引
                # 初始化当前路径的 SMILES 到 UUID 映射
                pathway_smiles_to_uuid = {}
                pathway_edges = []

                # 提取路径的边信息并构建 UDS 数据
                route_edges = []
                if isinstance(output_graph, MolSetGraph):
                    # MolSetGraph: 路径是有序节点序列，按深度提取相邻节点间的边
                    route_list = sorted(list(route), key=lambda x: x.depth)
                    for i in range(len(route_list) - 1):
                        u, v = route_list[i], route_list[i + 1]
                        edge_data = output_graph._graph.get_edge_data(u, v)
                    
                        node_depth = i+1
                    
                        if edge_data and "reaction" in edge_data:
                            reaction = edge_data["reaction"]
                            rxn_smiles = reaction.reaction_smiles
                            probability = reaction.metadata['probability']
                            template = edge_data['reaction'].metadata['template']
                            product = [*reaction.products]
                            reactants = [*reaction.reactants]

                            # 添加化学节点到 node_dict		
                            for chem_molecule in reactants:
                                chem_smiles = chem_molecule.smiles
                                chem_metadata = chem_molecule.metadata
                                if chem_smiles and chem_smiles not in uds["node_dict"]:
                                    # 从 metadata 中获取 purchasable 信息

                                    uds["node_dict"][chem_smiles] = {
                                        "smiles": chem_smiles,
                                        "as_reactant": template,
                                        "as_product": 1,
                                        "properties": [{"link": ""}, {"availability": ""}],
                                        "purchase_price": 1.0 if chem_metadata['is_purchasable'] else False,
                                        "terminal": node_depth > 0 and chem_metadata['is_purchasable'],
                                        "type": "chemical",
                                        "id": chem_smiles
                                    }

                                    # 生成/复用 UUID
                                    if chem_smiles == root_molecule.smiles:
                                        uuid = ROOT_UUID
                                    elif chem_smiles in global_smiles_to_uuid:
                                        uuid = global_smiles_to_uuid[chem_smiles]
                                    else:
                                        uuid = str(uuid4())
                                        global_smiles_to_uuid[chem_smiles] = uuid

                                    pathway_smiles_to_uuid[chem_smiles] = uuid
                                    if uuid not in uds["uuid2smiles"]:
                                        uds["uuid2smiles"][uuid] = chem_smiles

                            # 添加反应节点到 node_dict
                            if rxn_smiles and rxn_smiles not in uds["node_dict"]:
                                # 获取节点数据用于 model_metadata
                                node_data = {}
                                if hasattr(v, 'data'):
                                    node_data = dict(v.data)

                                uds["node_dict"][rxn_smiles] = {
                                    "smiles": rxn_smiles,
                                    "plausibility": probability,
                                    "rxn_score_from_model": probability,
                                    "model_metadata": [{
                                        "direction": "retro",
                                        "backend": "template_relevance",
                                        "model_name": "syntheseus",
                                        "attributes": {
                                            "max_num_templates": v.num_visit,
                                            "max_cum_prob": 0.999,
                                            "attribute_filter": []
                                        },
                                        "model_score": node_data['policy_score'],
                                        "normalized_model_score": node_data['policy_score'],
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
                                                "reaction_smarts": template,
                                                "template_set": "syntheseus",
                                                "_id": rxn_smiles,
                                                "template_score": node_data['policy_score'],
                                                "template_rank": 1,
                                                "num_examples": 1
                                            }
                                        }
                                    }],
                                    "precursor_properties": {
                                        "rms_molwt": 0.0,
                                        "num_rings": 0,
                                        "scscore": 0.0
                                    },
                                    "precursor_rank": 1,
                                    "precursor_score": probability,
                                    "reaction_properties": {
                                        "canonical_reaction_smiles": rxn_smiles,
                                        "mapped_smiles": rxn_smiles,
                                        "plausibility": probability,
                                        "reacting_atoms": [],
                                        "selec_error": None,
                                        "cluster_id": None,
                                        "cluster_name": None
                                    },
                                    "type": "reaction",
                                    "id": rxn_smiles
                                }

                                # 生成/复用反应 UUID
                                if rxn_smiles in global_smiles_to_uuid:
                                    uuid = global_smiles_to_uuid[rxn_smiles]
                                else:
                                    uuid = str(uuid4())
                                    global_smiles_to_uuid[rxn_smiles] = uuid
                                pathway_smiles_to_uuid[rxn_smiles] = uuid
                                if uuid not in uds["uuid2smiles"]:
                                    uds["uuid2smiles"][uuid] = rxn_smiles

                            # 添加到 graph（使用 SMILES）
                            # 产物 -> 反应
                            for prod_mol in product:
                                prod_smiles = prod_mol.smiles
                                if prod_smiles and rxn_smiles:
                                    edge_key = (prod_smiles, rxn_smiles)
                                    if not any(e.get('source') == prod_smiles and e.get('target') == rxn_smiles for e in uds["graph"]):
                                        uds["graph"].append({"source": prod_smiles, "target": rxn_smiles})

                            # 反应 -> 反应物
                            for reactant_mol in reactants:
                                reactant_smiles = reactant_mol.smiles
                                if reactant_smiles and rxn_smiles:
                                    edge_key = (rxn_smiles, reactant_smiles)
                                    if not any(e.get('source') == rxn_smiles and e.get('target') == reactant_smiles for e in uds["graph"]):
                                        uds["graph"].append({"source": rxn_smiles, "target": reactant_smiles})

                            # 确保 rxn_smiles 在 pathway_smiles_to_uuid 中
                            if rxn_smiles and rxn_smiles not in pathway_smiles_to_uuid:
                                if rxn_smiles in global_smiles_to_uuid:
                                    pathway_smiles_to_uuid[rxn_smiles] = global_smiles_to_uuid[rxn_smiles]
                                else:
                                    # 这种情况不应该发生，因为应该在前面已经处理过
                                    uuid = str(uuid4())
                                    global_smiles_to_uuid[rxn_smiles] = uuid
                                    pathway_smiles_to_uuid[rxn_smiles] = uuid

                            # 确保 product 的 SMILES 在 pathway_smiles_to_uuid 中
                            for prod_mol in product:
                                prod_smiles = prod_mol.smiles
                                if prod_smiles and prod_smiles not in pathway_smiles_to_uuid:
                                    if prod_smiles in global_smiles_to_uuid:
                                        pathway_smiles_to_uuid[prod_smiles] = global_smiles_to_uuid[prod_smiles]
                                    else:
                                        uuid = str(uuid4())
                                        global_smiles_to_uuid[prod_smiles] = uuid
                                        pathway_smiles_to_uuid[prod_smiles] = uuid

                            # 确保 reactants 的 SMILES 在 pathway_smiles_to_uuid 中
                            for reactant_mol in reactants:
                                reactant_smiles = reactant_mol.smiles
                                if reactant_smiles and reactant_smiles not in pathway_smiles_to_uuid:
                                    if reactant_smiles in global_smiles_to_uuid:
                                        pathway_smiles_to_uuid[reactant_smiles] = global_smiles_to_uuid[reactant_smiles]
                                    else:
                                        uuid = str(uuid4())
                                        global_smiles_to_uuid[reactant_smiles] = uuid
                                        pathway_smiles_to_uuid[reactant_smiles] = uuid

                            # 添加到 pathways（使用 UUID）
                            # 产物 -> 反应
                            for prod_mol in product:
                                prod_smiles = prod_mol.smiles
                                if prod_smiles and rxn_smiles:
                                    pathway_edges.append({
                                        "source": pathway_smiles_to_uuid[prod_smiles],
                                        "target": pathway_smiles_to_uuid[rxn_smiles]
                                    })

                            # 反应 -> 反应物
                            for reactant_mol in reactants:
                                reactant_smiles = reactant_mol.smiles
                                if reactant_smiles and rxn_smiles:
                                    pathway_edges.append({
                                        "source": pathway_smiles_to_uuid[rxn_smiles],
                                        "target": pathway_smiles_to_uuid[reactant_smiles]
                                    })

                            # 保留原有的 route_edges 格式
                            route_edges.append({
                                'source_id': str(id(u)),
                                'target_id': str(id(v)),
                                'reaction': {
                                    'reaction_smiles': rxn_smiles,
                                    'reactants': [r.smiles for r in reactants],
                                    'product': [p.smiles for p in product],
                                    'metadata': {'probability': probability,
                                                 'template': template}
                                }
                            })
                # 添加到 UDS pathways 和 pathways_properties
                if pathway_edges:
                    uds["pathways"].append(pathway_edges)
                    # 计算路径深度
                    max_depth = max(node.depth for node in route_list)
                    uds["pathways_properties"].append({
                        "depth": max_depth,
                        "precursor_cost": "precursor_cost",
                        "score": None,
                        "cluster_id": None
                    })

                # 单独可视化 route
                # 序列化节点（保留原有逻辑用于单独的 route 文件）
                serialized_nodes = []
                node_id_to_index = {}
                for idx, node in enumerate(route_list):
                    node_id = str(id(node))
                    node_id_to_index[node_id] = idx

                    # 手动序列化节点
                    serialized_node = {
                        'id': node_id,
                        'has_solution': node.has_solution,
                        'num_visit': node.num_visit,
                        'depth': node.depth,
                        'is_expanded': node.is_expanded,
                        'creation_time': node.creation_time.isoformat() if node.creation_time else None,
                        'data': dict(node.data),
                        'local_index': idx
                    }

                    if hasattr(node, 'mol'):
                        serialized_node.update({
                            'type': 'OrNode',
                            'smiles': node.mol.smiles,
                            'is_purchasable': node.mol.metadata.get('is_purchasable', False),
                        })
                    elif hasattr(node, 'reaction'):
                        serialized_node.update({
                            'type': 'AndNode',
                            'reaction_smiles': node.reaction.reaction_smiles,
                            'reactants': [r.smiles for r in node.reaction.reactants],
                            'product': node.reaction.product.smiles,
                        })
                    elif hasattr(node, 'mols'):
                        serialized_node.update({
                            'type': 'MolSetNode',
                            'smiles_list': sorted([mol.smiles for mol in node.mols]),
                            'is_purchasable': all(mol.metadata.get('is_purchasable', False) for mol in node.mols),
                        })

                    serialized_nodes.append(serialized_node)

                # 更新边中的索引引用
                for edge in route_edges:
                    edge['source_index'] = node_id_to_index[edge['source_id']]
                    edge['target_index'] = node_id_to_index[edge['target_id']]

                # 构建 JSON 路由数据
                route_data = {
                    'route_index': route_idx,
                    'graph_type': type(output_graph).__name__,
                    'nodes': serialized_nodes,
                    'edges': route_edges,
                    'num_nodes': len(serialized_nodes),
                    'num_edges': len(route_edges)
                }

                # 保存为 JSON（用于调试查看数据）
                with open(results_dir / f"route_{route_idx}.json", "w") as f_route:
                    json.dump(route_data, f_route, indent=2, ensure_ascii=False)

                visualize_kwargs: Dict[str, Any] = dict(
                    graph=output_graph,
                    filename=str(results_dir / f"route_{route_idx}.pdf"),
                    nodes=route,
                )

                if isinstance(output_graph, AndOrGraph):
                    visualize_andor(**visualize_kwargs)
                elif isinstance(output_graph, MolSetGraph):
                    visualize_molset(**visualize_kwargs)
                else:
                    assert False

            # 生成 UDS Askcos 格式输出
            if is_resume_mode:
                # 恢复模式：保存 new_uds.json 并合并到 uds_askcos.json
                new_uds_file = results_dir / "new_uds.json"
                with open(new_uds_file, "w") as f_uds:
                    json.dump(uds, f_uds, indent=2, ensure_ascii=False)
                logger.info(f"Saved {len(uds['pathways'])} new pathways to new_uds.json")

                # 合并新旧路径
                merged_uds = merge_uds_data(resume_old_uds, uds)
                uds_file = results_dir / "uds_askcos.json"
                with open(uds_file, "w") as f_uds:
                    json.dump(merged_uds, f_uds, indent=2, ensure_ascii=False)
                logger.info(f"Updated uds_askcos.json with {len(merged_uds['pathways'])} total pathways")
            else:
                # 正常模式：直接保存 uds_askcos.json
                uds_file = results_dir / "uds_askcos.json"
                with open(uds_file, "w") as f_uds:
                    json.dump(uds, f_uds, indent=2, ensure_ascii=False)
                logger.info(f"UDS Askcos format output saved to {uds_file}")

        results_lock_path.unlink()
        del results_dir

    if num_targets > 1:
        logger.info(f"Writing summary statistics across all {num_targets} targets")
        combined_stats: Dict[str, float] = dict(
            num_targets=num_targets,
            num_solved_targets=sum(stats["soln_time_wallclock"] != math.inf for stats in all_stats),
        )

        for key in [
            "rxn_model_calls_used",
            "num_nodes_in_final_tree",
            "soln_time_rxn_model_calls",
            "soln_time_wallclock",
        ]:
            values = [stats[key] for stats in all_stats]
            combined_stats[f"average_{key}"] = statistics.mean(values)
            combined_stats[f"median_{key}"] = statistics.median(values)

        logger.info(pformat(combined_stats))

        with open(results_dir_current_run / "stats.json", "wt") as f_combined_stats:
            f_combined_stats.write(json.dumps(combined_stats, indent=2))

    return results_dir_current_run


def main(argv: Optional[List[str]] = None, config_cls: Any = SearchConfig) -> Path:
    config = cli_get_config(argv=argv, config_cls=config_cls)

    def _warn_will_not_use_defaults(message: str) -> None:
        logger.warning(f"{message}; no model-specific search hyperparameters will be used")

    defaults_file_path = Path(__file__).parent / "search_config.yml"
    if not defaults_file_path.exists():
        _warn_will_not_use_defaults(f"File {defaults_file_path} does not exist")
    else:
        with open(defaults_file_path, "rt") as f_defaults:
            defaults = yaml.safe_load(f_defaults)

        search_algorithm_name = config.search_algorithm.name
        if search_algorithm_name not in defaults:
            _warn_will_not_use_defaults(
                f"Hyperparameter defaults file has no entry for {search_algorithm_name}"
            )
        else:
            search_algorithm_defaults = defaults[search_algorithm_name]

            model_name = config.model_class.name
            if model_name not in search_algorithm_defaults:
                _warn_will_not_use_defaults(
                    f"Hyperparameter defaults file has no entry for {model_name}"
                )
            else:
                relevant_defaults = search_algorithm_defaults[model_name]
                logger.info(
                    f"Using hyperparameter defaults from {defaults_file_path}: {relevant_defaults}"
                )

                # We now parse the config again (we could not have included the defaults earlier as
                # we did not know the search algorithm and model class before the first parsing).
                config = cli_get_config(
                    argv=argv,
                    config_cls=config_cls,
                    defaults={f"{search_algorithm_name}_config": relevant_defaults},
                )

    return run_from_config(config)


if __name__ == "__main__":
    import sys
    if len(sys.argv)>1:
        main()
    else:
        # search_target='Cc1c(C[C@H](C)N)sc2c(NCc3cccs3)cc(Cl)nc12'
        # search_target="NC1=Nc2ccc(F)cc2C2CCCC12"
        # "COc1cccc(OC(=O)/C=C/c2cc(OC)c(OC)c(OC)c2)c1"
        # CNC(=O)COc1cc(Cl)c(Cc2ccc(O)c(C(C)C)c2)c(Cl)c1
        argv = [
            "inventory_smiles_file=/home/liwenlong/chemTools/retro_syn/syntheseus/emolecules.txt",
            "search_target=C1=COC(/C=C2/C(=O)C3=C(C/2=O)C=CC=C3)=C1",
            "model_class=SimpRetro",
            "model_dir=/home/liwenlong/chemTools/retro_syn/syntheseus/syntheseus/SimpRetro_templates copy.json",
            "time_limit_s=40",
            "search_algorithm=mcts",
            "results_dir=retro_mcts_results/",
            "use_gpu=False",
            "num_routes_to_plot=50",
            "mcts_config.max_expansion_depth=20",
            "expand_purchasable_target=True",
            "resume_search=/home/liwenlong/retro_mcts_results/SimpRetro_2026-03-12T17:37:38"
        ]
        main(argv=argv)
        
