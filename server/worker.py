"""Worker process module: each worker process loads its own model + inventory,
then executes search requests independently."""

"""Worker process module: each worker process loads its own model + inventory,
then executes search requests independently."""

import logging
import os
import time
from typing import Any

from omegaconf import OmegaConf
from tqdm import tqdm

from syntheseus import Molecule
from syntheseus.reaction_prediction.inference.config import BackwardModelClass, BackwardModelConfig
from syntheseus.reaction_prediction.utils.model_loading import get_model
from syntheseus.reaction_prediction.utils.misc import set_random_seed
from syntheseus.search import INT_INF
from syntheseus.search.mol_inventory import SmilesListInventory

try:
    from server.schemas import SearchRequest
except ImportError:
    from schemas import SearchRequest

logger = logging.getLogger(__name__)

# Per-process globals: model and inventory are loaded once per worker.
_MODEL: Any = None
_INVENTORY: Any = None


def _build_model_config() -> BackwardModelConfig:
    """构建与 server.py 一致的模型配置。"""
    return BackwardModelConfig(
        model_class=BackwardModelClass.SimpRetro,
        model_dir=os.getenv(
            "MODEL_DIR",
            "/home/liwenlong/chemTools/retro_syn/syntheseus/syntheseus/SimpRetro_templates copy.json",
        ),
    )


def init_worker(inventory_smiles: set, canonicalize: bool = False) -> None:
    """在 worker 进程启动时调用。

    Fork 模式下：主进程已调用过此函数，子进程直接返回（继承全局变量）。
    Spawn 模式下：每个 worker 独立调用，各自加载模型和库存。

    Args:
        inventory_smiles: 主进程已加载的库存 SMILES 集合（spawn 模式下需独立加载则传空 set）
        canonicalize: 是否对库存做标准化
    """
    global _MODEL, _INVENTORY

    # Fork 模式下主进程已初始化，跳过
    if _MODEL is not None:
        logger.info("Worker: model already initialized (fork mode, inherited from parent)")
        return

    set_random_seed(0)

    # 1. 加载库存
    if inventory_smiles:
        # 复用主进程传入的库存
        logger.info("Worker: receiving inventory from parent process (%d molecules)", len(inventory_smiles))
        _INVENTORY = SmilesListInventory(list(inventory_smiles), canonicalize=canonicalize)
    else:
        # spawn 模式下从文件独立加载
        inventory_path = os.getenv(
            "INVENTORY_PATH",
            "/home/liwenlong/chemTools/retro_syn/syntheseus/emolecules.txt",
        )
        logger.info("Worker: loading inventory from %s ...", inventory_path)
        _INVENTORY = SmilesListInventory.load_from_file(inventory_path, canonicalize=canonicalize)
    logger.info("Worker: inventory ready (%d molecules)", len(_INVENTORY))

    # 2. 加载模型
    model_config = _build_model_config()
    config_dict = OmegaConf.structured(model_config)
    use_gpu = os.getenv("USE_GPU", "true").lower() in ("true", "1", "yes")

    logger.info("Worker: loading model from %s ...", model_config.model_dir)
    _MODEL = get_model(
        config_dict,
        batch_size=1,
        num_gpus=1 if use_gpu else 0,
        use_cache=True,
        default_num_results=50,
        inventory_file=_INVENTORY._smiles_set,
    )
    logger.info("Worker: model loaded, ready for search")


def run_search(req: SearchRequest) -> dict:
    """在工作进程中执行一次搜索。

    Args:
        req: 搜索请求

    Returns:
        搜索结果的 dict 表示（Pydantic model 序列化）
    """
    from syntheseus.search.algorithms.best_first.retro_star import RetroStarSearch
    from syntheseus.search.algorithms.mcts.base import pucb_bound
    from syntheseus.search.algorithms.mcts.molset import MolSetMCTS
    from syntheseus.search.algorithms.pdvn import PDVN_MCTS
    from syntheseus.search.analysis.route_extraction import iter_routes_cost_order
    from syntheseus.search.node_evaluation.common import (
        ConstantNodeEvaluator,
        HasSolutionValueFunction,
        ReactionModelLogProbCost,
        ReactionModelProbPolicy,
    )

    if _MODEL is None or _INVENTORY is None:
        raise RuntimeError("Worker not initialized: call init_worker() first")

    start = time.time()
    target = Molecule(req.smiles)

    # Build algorithm (对齐 engine.py _build_algorithm)
    bto = req.build_tree_options
    kwargs: dict = {
        "reaction_model": _MODEL,
        "mol_inventory": _INVENTORY,
        "time_limit_s": bto.expansion_time,
        "limit_reaction_model_calls": 1_000_000,
        "limit_iterations": 1_000_000,
        "limit_graph_nodes": INT_INF,
        "prevent_repeat_mol_in_trees": True,
        "stop_on_first_solution": bto.stop_on_first_solution,
        "expand_purchasable_target": bto.expand_purchasable_target,
    }

    backend = req.backend.lower()

    if backend == "mcts":
        kwargs.update({
            "value_function": ConstantNodeEvaluator(0.5),
            "reward_function": HasSolutionValueFunction(),
            "policy": ReactionModelProbPolicy(),
            "bound_constant": 1.0,
            "bound_function": pucb_bound,
            "max_expansion_depth": 20,
        })
        alg = MolSetMCTS(**kwargs)
    elif backend == "retro_star":
        kwargs.update({
            "value_function": ConstantNodeEvaluator(0.0),
            "and_node_cost_fn": ReactionModelLogProbCost(),
        })
        alg = RetroStarSearch(**kwargs)
    elif backend == "pdvn":
        kwargs.update({
            "value_function_syn": ConstantNodeEvaluator(0.5),
            "value_function_cost": ConstantNodeEvaluator(0.0),
            "and_node_cost_fn": ConstantNodeEvaluator(0.1),
            "policy": ReactionModelProbPolicy(),
            "bound_constant": 100.0,
            "bound_function": pucb_bound,
            "c_dead": 5.0,
        })
        alg = PDVN_MCTS(**kwargs)
    else:
        raise ValueError(f"Unsupported backend: {backend!r}. Supported: mcts, retro_star, pdvn")

    alg.reset()
    graph, _ = alg.run_from_mol(target)
    elapsed = time.time() - start

    # 用 tqdm 打印搜索耗时摘要
    tqdm.write(
        f"搜索 [{target.smiles}]  实际耗时={elapsed:.1f}s / 限制={bto.expansion_time:.0f}s  "
        f"nodes={len(graph)}  rxn_calls={alg.reaction_model.num_calls()}  "
        f"solution={'YES' if graph.root_node.has_solution else 'NO'}"
    )

    # Extract routes (对齐 engine.py _extract_routes)
    max_branching = bto.max_branching
    routes = []
    if graph.root_node.has_solution:
        for idx, route_nodes in enumerate(iter_routes_cost_order(graph, max_routes=max_branching)):
            try:
                synth_graph = graph.to_synthesis_graph(nodes=route_nodes)
            except AssertionError as e:
                logger.warning("Skipping invalid route: %s", e)
                continue

            reactions = []
            total_score = 0.0
            for rxn in synth_graph.nodes():
                score = rxn.metadata.get("score", 0.0)
                reactions.append({
                    "reaction_smiles": rxn.reaction_smiles,
                    "score": score,
                })
                total_score += score

            routes.append({
                "route_index": idx,
                "reactions": reactions,
                "total_score": total_score,
            })

    return {
        "target_smiles": target.smiles,
        "routes": routes,
        "num_routes_found": len(routes),
        "time_elapsed_s": round(elapsed, 3),
        "graph_nodes": len(graph),
    }
