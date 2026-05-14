import logging
import time
from typing import Any

from syntheseus import Molecule
from syntheseus.reaction_prediction.utils.model_loading import get_model
from syntheseus.search import INT_INF
from syntheseus.search.analysis.route_extraction import iter_routes_cost_order
from syntheseus.search.algorithms.best_first.retro_star import RetroStarSearch
from syntheseus.search.algorithms.mcts.base import pucb_bound
from syntheseus.search.algorithms.mcts.molset import MolSetMCTS
from syntheseus.search.algorithms.pdvn import PDVN_MCTS
from syntheseus.search.graph.and_or import AndOrGraph
from syntheseus.search.graph.molset import MolSetGraph
from syntheseus.search.mol_inventory import SmilesListInventory
from syntheseus.search.node_evaluation.common import (
    ConstantNodeEvaluator,
    HasSolutionValueFunction,
    ReactionModelLogProbCost,
    ReactionModelProbPolicy,
)

from server.schemas import ReactionStep, RouteInfo, SearchRequest, SearchResponse

logger = logging.getLogger(__name__)


class SearchEngine:
    """逆合成搜索引擎，持有预加载的模型和库存。"""

    def __init__(self, model: Any, inventory: SmilesListInventory) -> None:
        self.model = model
        self.inventory = inventory

    def search(self, req: SearchRequest) -> SearchResponse:
        """执行一次逆合成搜索，返回路线列表。"""
        start = time.time()
        target = Molecule(req.smiles)
        alg = self._build_algorithm(req)
        alg.reset()
        graph, _ = alg.run_from_mol(target)
        elapsed = time.time() - start

        routes = self._extract_routes(graph, req.build_tree_options.max_branching)

        return SearchResponse(
            target_smiles=target.smiles,
            routes=routes,
            num_routes_found=len(routes),
            time_elapsed_s=round(elapsed, 3),
            graph_nodes=len(graph),
        )

    def _build_algorithm(self, req: SearchRequest):
        """根据 backend 创建搜索算法实例。对齐 search.py 的算法构建逻辑。"""
        # 对齐 search.py search_algorithm_config_to_kwargs() 提取的全部参数
        # search.py 第 196-207 行
        bto = req.build_tree_options
        kwargs: dict = {
            "reaction_model": self.model,
            "mol_inventory": self.inventory,
            # 公共终止条件（对齐 search.py 第 197-207 行）
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
            # 对齐 search.py MCTS 分支 + search_config.yml LocalRetro/SimpRetro 默认
            kwargs.update({
                "value_function": ConstantNodeEvaluator(0.5),
                "reward_function": HasSolutionValueFunction(),
                "policy": ReactionModelProbPolicy(),
                "bound_constant": 1.0,
                "bound_function": pucb_bound,
                "max_expansion_depth": 20,
            })
            return MolSetMCTS(**kwargs)

        elif backend == "retro_star":
            # 对齐 search.py retro_star 分支
            kwargs.update({
                "value_function": ConstantNodeEvaluator(0.0),
                "and_node_cost_fn": ReactionModelLogProbCost(),
            })
            return RetroStarSearch(**kwargs)

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
            return PDVN_MCTS(**kwargs)

        else:
            raise ValueError(
                f"Unsupported backend: {backend!r}. "
                f"Supported: mcts, retro_star, pdvn"
            )

    def _extract_routes(self, graph, max_routes: int) -> list[RouteInfo]:
        """从搜索图中提取路线，对齐 search.py 第 576-604 行逻辑。"""
        if not graph.root_node.has_solution:
            logger.info("Search found no solution for the target molecule")
            return []

        routes = []
        for idx, route_nodes in enumerate(
            iter_routes_cost_order(graph, max_routes=max_routes)
        ):
            # 安全提取路线：根节点必须至少有一个子反应
            try:
                synth_graph = graph.to_synthesis_graph(nodes=route_nodes)
            except AssertionError as e:
                logger.warning("Skipping invalid route: %s", e)
                continue

            reactions = []
            total_score = 0.0
            for rxn in synth_graph.nodes():
                score = rxn.metadata.get("score", 0.0)
                reactions.append(ReactionStep(
                    reaction_smiles=rxn.reaction_smiles,
                    score=score,
                ))
                total_score += score

            routes.append(RouteInfo(
                route_index=idx,
                reactions=reactions,
                total_score=total_score,
            ))
        return routes
