from typing import List, Optional

from pydantic import BaseModel, Field


class BuildTreeOptions(BaseModel):
    expansion_time: float = Field(default=30.0, description="搜索时间上限（秒）")
    max_branching: int = Field(default=50, description="最多返回路线数")
    max_iterations: int = Field(default=0, description="0 = 不限制迭代次数")
    expand_purchasable_target: bool = Field(default=True, description="即使目标分子在库存中也展开搜索")
    stop_on_first_solution: bool = Field(default=False, description="找到第一个解即停止")


class SearchRequest(BaseModel):
    smiles: str = Field(..., description="目标分子 SMILES")
    backend: str = Field(default="mcts", description="搜索算法：mcts / retro_star / pdvn")
    build_tree_options: BuildTreeOptions = Field(default_factory=BuildTreeOptions)
    timeout: int = Field(default=300, description="请求超时（秒）")


class ReactionStep(BaseModel):
    reaction_smiles: str
    score: float


class RouteInfo(BaseModel):
    route_index: int
    reactions: List[ReactionStep]
    total_score: float


class SearchResponse(BaseModel):
    target_smiles: str
    routes: List[RouteInfo]
    num_routes_found: int
    time_elapsed_s: float
    graph_nodes: int
