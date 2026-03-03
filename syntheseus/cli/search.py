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
from syntheseus.search.analysis.route_extraction import iter_routes_time_order
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


class RouteTracker:
    """追踪已输出的路径，避免重复显示."""

    def __init__(self) -> None:
        self.shown_route_hashes: set[str] = set()

    def _hash_route(self, route: set) -> str:
        """计算路径的哈希值用于去重."""
        # 使用节点ID的排序元组作为哈希
        node_ids = sorted([id(node) for node in route])
        return str(tuple(node_ids))

    def get_new_routes(
        self,
        graph: AndOrGraph | MolSetGraph,
        max_routes: int,
    ) -> list[set]:
        """获取新发现的路径.

        Args:
            graph: 搜索图
            max_routes: 最大返回路径数

        Returns:
            新发现的路径列表（每个路径是一个节点集合）
        """
        new_routes = []

        for route in iter_routes_time_order(graph, max_routes=max_routes * 2):
            # 这里我们多获取一些路径，然后过滤出新的
            route_hash = self._hash_route(route)

            if route_hash not in self.shown_route_hashes:
                new_routes.append(route)
                self.shown_route_hashes.add(route_hash)

                if len(new_routes) >= max_routes:
                    break

        return new_routes


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


def extract_and_plot_routes(
    graph: AndOrGraph | MolSetGraph,
    results_dir: Path,
    route_tracker: RouteTracker,
    max_routes: int,
    iteration: int,
) -> int:
    """提取并绘制路径（仅新路径）.

    Args:
        graph: 搜索图
        results_dir: 结果目录
        route_tracker: 路径追踪器
        max_routes: 最大绘制路径数
        iteration: 当前迭代次数

    Returns:
        新发现的路径数量
    """
    if not VISUALIZATION_CODE_IMPORTED:
        logger.warning("可视化代码未导入，跳过路径绘制")
        return 0

    # 获取新路径
    new_routes = route_tracker.get_new_routes(graph, max_routes)

    for route_idx, route in enumerate(new_routes):
        route_file_idx = iteration * max_routes + route_idx
        with open(results_dir / f"route_{route_file_idx}.pkl", "wb") as f_route:
            pickle.dump(route, f_route)

        visualize_kwargs: Dict[str, Any] = dict(
            graph=graph,
            filename=str(results_dir / f"route_{route_file_idx}.pdf"),
            nodes=route,
        )

        if isinstance(graph, AndOrGraph):
            visualize_andor(**visualize_kwargs)
        elif isinstance(graph, MolSetGraph):
            visualize_molset(**visualize_kwargs)
        else:
            assert False

    return len(new_routes)


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
    expand_purchasable_target: bool = True  # Whether to expand target even if it's purchasable


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

    # Fields configuring interactive incremental search mode
    interactive_mode: bool = False  # Whether to enable interactive incremental search
    increment_time_s: float = 30.0  # Time per search increment in interactive mode
    max_continues: int = 10  # Maximum number of continue prompts in interactive mode


@dataclass
class SearchConfig(BackwardModelConfig, BaseSearchConfig):
    """Config for running search for given search targets."""

    pass


def run_from_config(config: SearchConfig) -> Path:
    set_random_seed(0)

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

    # Prepare the output directory
    results_dir_top_level = Path(config.results_dir)

    dirname = config.model_class.name
    if config.append_timestamp_to_dir:
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        dirname += f"_{str(timestamp)}"

    results_dir_current_run = results_dir_top_level / dirname

    logger.info("Setup completed")
    num_targets = len(search_targets)

    all_stats: List[Dict[str, Any]] = []
    for idx, smiles in enumerate(tqdm(search_targets)):
        logger.info(f"Running search for target {smiles}")

        if num_targets == 1:
            results_dir = results_dir_current_run
        else:
            results_dir = results_dir_current_run / str(idx)

        results_dir.mkdir(parents=True, exist_ok=True)
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

        # ============================================================
        # 搜索执行：支持交互式增量搜索模式
        # ============================================================
        if config.interactive_mode:
            # 交互模式：增量搜索，每次 increment_time_s 秒
            increment_time_s = config.increment_time_s
            max_continues = config.max_continues

            alg.reset()
            output_graph = None
            total_search_time = 0.0
            route_tracker = RouteTracker()

            for iteration in range(max_continues):
                # 设置本次搜索的时间限制
                original_time_limit = alg.time_limit_s
                alg.time_limit_s = increment_time_s

                if output_graph is None:
                    # 第一次搜索：从分子开始
                    output_graph, _ = alg.run_from_mol(target)
                else:
                    # 后续搜索：在已有图上继续
                    _, _ = alg.run_from_graph(output_graph)

                # 恢复原始时间限制
                alg.time_limit_s = original_time_limit

                total_search_time += increment_time_s

                # ===== 完整输出（交互模式下） =====
                print(f"\n{'='*60}")
                print(f"已搜索 {total_search_time:.1f} 秒 (第 {iteration+1} 轮)")
                print(f"{'='*60}")

                # 1. 统计信息
                print_interim_stats(
                    output_graph, iteration, total_search_time,
                    alg.reaction_model, target, mol_inventory
                )

                # 2. 提取并绘制新发现的路径
                if config.num_routes_to_plot > 0:
                    new_routes_count = extract_and_plot_routes(
                        output_graph, results_dir, route_tracker,
                        config.num_routes_to_plot, iteration
                    )
                    print(f"本轮新发现路径: {new_routes_count} 条")

                # 3. 询问是否继续
                if iteration < max_continues - 1:
                    try:
                        user_input = input("\n是否继续搜索 {:.0f} 秒？(y/n): ".format(increment_time_s)).strip().lower()
                        if user_input != 'y':
                            print(f"\n搜索结束，总用时: {total_search_time:.1f} 秒")
                            break
                    except (EOFError, KeyboardInterrupt):
                        # 处理用户中断（Ctrl+C 或 Ctrl+D）
                        print(f"\n\n用户中断，搜索结束。总用时: {total_search_time:.1f} 秒")
                        break
            # 交互模式结束
            logger.info(f"Finished interactive search for target {smiles}, total time: {total_search_time:.1f}s")
        else:
            # 非交互模式：保持原有行为
            alg.reset()
            output_graph, _ = alg.run_from_mol(target)
            logger.info(f"Finished search for target {smiles}")

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
            "index": idx,
            "smiles": smiles,
            "target_in_inventory": target_in_inventory,
            "rxn_model_calls_used": alg.reaction_model.num_calls(),
            "num_nodes_in_final_tree": len(output_graph),
            "soln_time_rxn_model_calls": soln_time_rxn_model_calls,
            "soln_time_wallclock": soln_time_wallclock,
        }

        all_stats.append(stats)
        logger.info(pformat(stats))

        with open(results_stats_path, "wt") as f_stats:
            f_stats.write(json.dumps(stats, indent=2))

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

        if config.num_routes_to_plot > 0:
            # Extract some synthesis routes in the order they were found
            logger.info(f"Extracting up to {config.num_routes_to_plot} routes for analysis")

            # TODO(kmaziarz): Add options to extract a diverse (or otherwise interesting) subset.
            routes: Iterator = iter_routes_time_order(
                output_graph, max_routes=config.num_routes_to_plot
            )

            for route_idx, route in enumerate(routes):
                # 提取路径的边信息（对于 MolSetGraph，反应存储在边上）
                route_edges = []
                if isinstance(output_graph, MolSetGraph):
                    # MolSetGraph: 路径是有序节点序列，提取相邻节点间的边
                    route_list = list(route)
                    for i in range(len(route_list) - 1):
                        u, v = route_list[i], route_list[i + 1]
                        edge_data = output_graph._graph.get_edge_data(u, v)
                        if edge_data and "reaction" in edge_data:
                            route_edges.append({
                                'source_id': str(id(u)),
                                'target_id': str(id(v)),
                                'reaction': serialize_reaction(edge_data["reaction"])
                            })

                # 序列化节点
                serialized_nodes = []
                node_id_to_index = {}
                for idx, node in enumerate(route):
                    node_id = str(id(node))
                    node_id_to_index[node_id] = idx
                    serialized_node = serialize_node(node)
                    serialized_node['local_index'] = idx
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

                # 保存为 JSON（用于前端展示）
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

            # 生成 UDS 格式输出（与 Askcos 前端兼容）
            if config.save_graph:
                from uds_converter import convert_graph_to_uds

                # 重新获取路径迭代器
                routes_for_uds = iter_routes_time_order(
                    output_graph, max_routes=config.num_routes_to_plot
                )

                # 转换为 UDS 格式
                uds_data = convert_graph_to_uds(
                    output_graph=output_graph,
                    routes_iterator=routes_for_uds,
                    stats=stats,
                    max_routes=config.num_routes_to_plot
                )

                # 保存 UDS 格式文件
                with open(results_dir / "uds_output.json", "w") as f_uds:
                    json.dump(uds_data, f_uds, indent=2, ensure_ascii=False)
                logger.info(f"UDS format output saved to {results_dir / 'uds_output.json'}")

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
        argv = [
            "inventory_smiles_file=/home/liwenlong/chemTools/retro_syn/syntheseus/emolecules.txt",
            "search_target=C1C(=CC=C(C=1)I)CC(=O)OC",
            "model_class=SimpRetro",
            "model_dir=/home/liwenlong/chemTools/retro_syn/syntheseus/syntheseus/SimpRetro_templates copy.json",
            "time_limit_s=30",
            "search_algorithm=mcts",
            "results_dir=retro_mcts_results/",
            "use_gpu=False",
            "num_routes_to_plot=50",
        ]
        main(argv=argv)
        
