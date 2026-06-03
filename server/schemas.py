from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class BuildTreeOptions(BaseModel):
    """搜索树构建参数。"""

    model_config = ConfigDict(json_schema_extra={
        "examples": [{
            "expansion_time": 30.0,
            "max_branching": 50,
            "save_graph": False,
            "resume_from": None,
        }]
    })

    expansion_time: float = Field(default=30.0, description="搜索时间上限（秒）")
    max_branching: int = Field(default=50, ge=1, le=500, description="最多返回路线数")
    max_iterations: int = Field(default=0, ge=0, description="迭代次数上限，0 = 不限制")
    expand_purchasable_target: bool = Field(default=True, description="即使目标分子在库存中也展开搜索")
    stop_on_first_solution: bool = Field(default=False, description="找到第一个解即停止")
    save_graph: bool = Field(default=False, description="缓存搜索图到磁盘（用于续算），返回 graph_id")
    resume_from: Optional[str] = Field(default=None, description="从指定 graph_id 的搜索图续算（需 save_graph 先启用）")


class SearchRequest(BaseModel):
    """逆合成搜索请求。"""

    model_config = ConfigDict(json_schema_extra={
        "examples": [{
            "smiles": "c1ccc(-c2ccccc2)cc1",
            "backend": "mcts",
            "build_tree_options": {"expansion_time": 30.0, "max_branching": 10},
        }]
    })

    smiles: str = Field(..., description="目标分子 SMILES", examples=["c1ccc(-c2ccccc2)cc1"])
    backend: str = Field(default="mcts", description="搜索算法", examples=["mcts", "retro_star", "pdvn"])
    build_tree_options: BuildTreeOptions = Field(default_factory=BuildTreeOptions)


class ReactionStep(BaseModel):
    """合成路线中的单步反应。"""

    reaction_smiles: str = Field(..., description="反应 SMILES（反应物.反应物>>产物）")
    score: float = Field(..., description="反应模型给出的评分")


class RouteInfo(BaseModel):
    """一条完整的合成路线。"""

    route_index: int = Field(..., description="路线序号（按成本排序）")
    reactions: List[ReactionStep] = Field(..., description="路线中的反应步骤列表")
    total_score: float = Field(..., description="路线总分（各步反应 score 之和）")


class SearchResponse(BaseModel):
    """逆合成搜索结果。"""

    target_smiles: str = Field(..., description="目标分子 SMILES")
    routes: List[RouteInfo] = Field(..., description="找到的合成路线列表")
    num_routes_found: int = Field(..., description="找到的路线总数")
    time_elapsed_s: float = Field(..., description="搜索耗时（秒）")
    graph_nodes: int = Field(..., description="搜索图节点总数")
    rxn_calls: int = Field(..., description="反应模型调用次数")
    graph_id: Optional[str] = Field(default=None, description="缓存的搜索图 ID，可传入 resume_from 进行续算")


class TaskSubmitResponse(BaseModel):
    """异步任务提交响应。"""

    task_id: str = Field(..., description="任务 ID，用于轮询状态")
    status: str = Field(..., description="任务状态", examples=["pending", "completed"])


class TaskStatusResponse(BaseModel):
    """异步任务状态查询响应。"""

    task_id: str = Field(..., description="任务 ID")
    status: str = Field(..., description="任务状态: pending / running / completed / failed")
    result: Optional[SearchResponse] = Field(default=None, description="搜索结果（status=completed 时有值）")
    error: Optional[str] = Field(default=None, description="错误信息（status=failed 时有值）")
