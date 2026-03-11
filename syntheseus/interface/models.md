# models.py 文件详细讲解

> 目标读者：高中生背景，想要深入理解代码

## 一、文件概述

这个文件定义了化学反应预测系统的**核心基类**。可以把它想象成一个"蓝图"或"模板"，所有具体的反应模型（比如预测逆合成反应的模型）都必须遵循这个蓝图。

---

## 二、重要概念解释

### 2.1 什么是"泛型"（Generic）？

在第24行：
```python
class ReactionModel(Generic[InputType, ReactionType]):
```

这就像一个**带参数的模板**：
- `InputType`：输入类型（可以是分子、或一组分子）
- `ReactionType`：反应类型（不同方向的反应）

类比：这就像蛋糕模具，你可以放入不同的原料（输入），得到不同的蛋糕（反应）。

### 2.2 什么是"抽象方法"（@abstractmethod）？

在第153行：
```python
@abstractmethod
def _get_reactions(...):
```

这是一个**必须被重写的方法**。父类只说"你需要有这个方法"，但不提供具体实现。子类必须自己写具体的代码。

类比：就像作业要求"写一篇作文"，老师只要求你写，但不告诉你具体写什么内容。

---

## 三、ReactionModel 类的核心：_get_reactions 方法

### 3.1 方法签名（第154-156行）

```python
def _get_reactions(
    self, inputs: list[InputType], num_results: int
) -> list[Sequence[ReactionType]]:
```

**逐行解释：**

- `def _get_reactions(`：定义一个方法，以下划线开头表示这是"内部方法"
- `self,`：表示这是类的方法，self代表对象本身
- `inputs: list[InputType],`：输入是一个列表，列表中的元素类型是 InputType
  - 对于逆合成反应：InputType 就是 Molecule（单个分子）
  - 对于正向反应：InputType 可能是 Bag[Molecule]（多个分子）
- `num_results: int`：需要返回多少个结果
- `) -> list[Sequence[ReactionType]]:`：返回值是一个列表，每个元素是一组反应

**举例说明：**
```python
# 假设我们预测逆合成反应（倒推合成路径）
inputs = [分子A, 分子B, 分子C]  # 3个目标分子
num_results = 10  # 每个分子返回10种可能的反应

# 返回值可能是：
[
    [反应1, 反应2, ..., 反应10],  # 分子A的10种可能的逆反应
    [反应1, 反应2, ..., 反应10],  # 分子B的10种可能的逆反应
    [反应1, 反应2, ..., 反应10],  # 分子C的10种可能的逆反应
]
```

### 3.2 这个方法的作用

这是一个**抽象方法**，意味着：
1. 父类不提供具体实现
2. 子类**必须**实现这个方法
3. 不同子类可以有不同的实现方式

**为什么这样设计？**
因为不同的预测模型有不同的实现方式：
- 有的调用神经网络模型
- 有的调用基于规则的系统
- 有的调用远程API

但它们都遵循相同的接口：输入分子列表 → 返回反应列表

---

## 四、ReactionModel 类详细讲解

### 4.1 初始化方法 __init__（第27-61行）

```python
def __init__(
    self,
    *,
    remove_duplicates: bool = True,
    use_cache: bool = False,
    count_cache_in_num_calls: bool = False,
    initial_cache: Optional[dict[tuple[InputType, int], Sequence[ReactionType]]] = None,
    max_cache_size: Optional[int] = None,
    default_num_results: int = DEFAULT_NUM_RESULTS,
    **kwargs,
) -> None:
```

**参数详细解释：**

| 参数名 | 类型 | 默认值 | 作用 |
|--------|------|--------|------|
| `remove_duplicates` | bool | True | 是否去除重复的反应 |
| `use_cache` | bool | False | 是否使用缓存（记住之前的结果） |
| `count_cache_in_num_calls` | bool | False | 缓存命中是否算作一次调用 |
| `initial_cache` | dict 或 None | None | 初始缓存内容 |
| `max_cache_size` | int 或 None | None | 缓存最大容量 |
| `default_num_results` | int | 100 | 默认返回多少个结果 |

**什么是缓存（Cache）？**

缓存就像一个小笔记本，记录之前计算过的结果：

```python
# 第一次：调用模型（慢，需要计算）
输入: 分子A → 模型计算 → 输出: [反应1, 反应2, ...]  # 存入缓存

# 第二次：从缓存读取（快，不需要计算）
输入: 分子A → 直接返回缓存的结果 → [反应1, 反应2, ...]
```

**代码执行流程：**

```python
# 第38行：调用父类的初始化
super().__init__(**kwargs)

# 第39-40行：保存配置
self.count_cache_in_num_calls = count_cache_in_num_calls
self.default_num_results = default_num_results

# 第44-47行：初始化缓存相关的属性
self._use_cache = False  # 是否启用缓存
self._cache = OrderedDict()  # 缓存本身（有序字典）
self._max_cache_size = max_cache_size  # 最大缓存大小
self._remove_duplicates = remove_duplicates  # 是否去重

# 第48行：重置缓存
self.reset(use_cache=use_cache)

# 第51-61行：如果提供了初始缓存，加载它
if initial_cache is not None:
    if self._use_cache:
        # 检查初始缓存是否超过最大大小
        if self._max_cache_size is not None and len(initial_cache) > self._max_cache_size:
            raise ValueError("Initial cache size exceeds `max_cache_size`.")
        self._cache.update(initial_cache)  # 添加到缓存
    else:
        warnings.warn("Initial cache was provided but will be ignored...")
```

### 4.2 reset 方法（第63-75行）

```python
def reset(self, use_cache: Optional[bool] = None) -> None:
    """Reset counts, caches, etc for this model."""
    self._cache.clear()  # 清空缓存

    # 如果指定了新的 use_cache 值，更新它
    if use_cache is not None:
        self._use_cache = use_cache

    # 重置缓存命中/未命中计数
    self._num_cache_hits = 0
    self._num_cache_misses = 0
```

**为什么要 reset？**
- 开始新的计算时，清空旧数据
- 重新开始统计

### 4.3 num_calls 方法（第77-97行）

```python
def num_calls(self, count_cache: Optional[bool] = None) -> int:
    """
    Number of times this reaction model has been called.

    Args:
        count_cache: if true, all calls are counted
            (even those retrieved from cache)
                If false, only count calls not in cache
    """
    if count_cache is None:
        count_cache = self.count_cache_in_num_calls

    if count_cache:
        return self._num_cache_hits + self._num_cache_misses
    else:
        return self._num_cache_misses
```

**统计意义：**
- `_num_cache_hits`：从缓存中获取的次数（快速）
- `_num_cache_misses`：需要实际计算的次数（慢）

### 4.4 __call__ 方法（第104-151行）⭐核心方法

这是**最重要的方法**！当你调用模型时，实际上就是调用这个方法。

```python
def __call__(
    self, inputs: list[InputType], num_results: Optional[int] = None
) -> list[Sequence[ReactionType]]:
```

**Python 特殊方法说明：**
`__call__` 是一个特殊方法，定义后，对象可以像函数一样被调用：

```python
model = ReactionModel()
# 这两种方式是等价的：
result = model.__call__(inputs, num_results=10)
result = model(inputs, num_results=10)  # 更简洁的写法
```

**方法执行流程（分步详解）：**

#### Step 0: 设置默认返回数量（第119行）
```python
num_results = num_results or self.default_num_results
```
如果没指定返回多少结果，就用默认值（100）

#### Step 1: 处理不在缓存中的输入（第122-130行）
```python
# 找出哪些输入不在缓存中
inputs_not_in_cache = deduplicate_keeping_order(
    [inp for inp in inputs if (inp, num_results) not in self._cache]
)

# 如果有需要计算的输入
if len(inputs_not_in_cache) > 0:
    # 调用子类实现的 _get_reactions 获取反应
    new_rxns = self._get_reactions(inputs=inputs_not_in_cache, num_results=num_results)
    assert len(new_rxns) == len(inputs_not_in_cache)

    # 存入缓存
    for inp, rxns in zip(inputs_not_in_cache, new_rxns):
        self._cache[(inp, num_results)] = self.filter_reactions(rxns)
```

**图解流程：**
```
输入: [分子A, 分子B, 分子C, 分子A]
       ↓
检查缓存:
  - 分子A: 在缓存 ✓
  - 分子B: 不在缓存 ✗
  - 分子C: 不在缓存 ✗
  - 分子A: 在缓存 ✓
       ↓
需要计算: [分子B, 分子C]
       ↓
调用 _get_reactions([分子B, 分子C])
       ↓
存入缓存，标记为"最近使用"
```

#### Step 2: 从缓存组装输出（第133-137行）
```python
output = []
for inp in inputs:
    key = (inp, num_results)
    output.append(self._cache[key])
    self._cache.move_to_end(key)  # 标记为最近使用
```

**LRU (Least Recently Used) 缓存策略：**
- 每次访问一个缓存项，把它移到末尾
- 最久未使用的项在开头
- 当缓存满时，删除开头的项

#### Step 2.1: 缓存管理（第139-145行）
```python
# 如果不使用缓存，清空
if not self._use_cache:
    self._cache.clear()
# 如果设置了最大缓存大小，删除多余的旧项
elif self._max_cache_size is not None:
    while len(self._cache) > self._max_cache_size:
        self._cache.popitem(last=False)  # 删除最旧的项
```

#### Step 3: 更新统计（第148-149行）
```python
self._num_cache_misses += len(inputs_not_in_cache)
self._num_cache_hits += len(inputs) - len(inputs_not_in_cache)
```

### 4.5 filter_reactions 方法（第163-171行）

```python
def filter_reactions(self, reaction_list: Sequence[ReactionType]) -> Sequence[ReactionType]:
    """Filters a list of reactions."""
    if self._remove_duplicates:
        return deduplicate_keeping_order(reaction_list)
    else:
        return list(reaction_list)
```

**作用：** 去除重复的反应

### 4.6 其他辅助方法

```python
def is_forward(self) -> bool:  # 是否是正向反应模型
    pass

def is_backward(self) -> bool:  # 是否是逆向反应模型
    return not self.is_forward()
```

---

## 五、两个重要的子类

### 5.1 BackwardReactionModel（第196-198行）

```python
class BackwardReactionModel(ReactionModel[Molecule, SingleProductReaction]):
    def is_forward(self) -> bool:
        return False
```

**用途：** 逆合成反应预测（从产品倒推原料）

**类型参数：**
- `InputType = Molecule`：输入是单个分子
- `ReactionType = SingleProductReaction`：反应类型是单产品反应

### 5.2 ForwardReactionModel（第201-203行）

```python
class ForwardReactionModel(ReactionModel[Bag[Molecule], Reaction]):
    def is_forward(self) -> bool:
        return True
```

**用途：** 正向反应预测（从原料预测产品）

**类型参数：**
- `InputType = Bag[Molecule]`：输入是一组分子（原料）
- `ReactionType = Reaction`：通用反应类型

---

## 六、完整流程示例

假设我们有一个逆合成反应模型：

```python
# 1. 创建模型
model = BackwardReactionModel(
    use_cache=True,
    default_num_results=10
)

# 2. 调用模型
molecules = [mol1, mol2, mol3]
results = model(molecules, num_results=10)

# 这实际执行的是：
results = model.__call__(molecules, num_results=10)
```

**内部发生了什么：**

```
__call__ 被调用
    ↓
检查缓存，找出需要计算的分子
    ↓
调用 _get_reactions (子类实现的方法)
    ↓
    例如: 调用神经网络模型预测
    返回: [[反应列表1], [反应列表2], [反应列表3]]
    ↓
过滤反应（去重）
    ↓
存入缓存
    ↓
组装输出
    ↓
更新统计
    ↓
返回结果
```

---

## 七、与用户选中的代码关系：Simpretro 的 _get_reactions 实现

### 7.1 继承关系

```
ReactionModel (models.py 中的基类)
    ↑
    |
ExternalBackwardReactionModel (中间抽象类)
    ↑
    |
SimpRetroModel (simpretro.py 中的具体实现)
```

### 7.2 _get_reactions 方法签名对比

**基类中的抽象定义 (models.py:153-161):**
```python
@abstractmethod
def _get_reactions(
    self, inputs: list[InputType], num_results: int
) -> list[Sequence[ReactionType]]:
    """
    Method to override which returns the underlying reactions.
    """
```

**SimpRetroModel 中的具体实现 (simpretro.py:193-294):**
```python
def _get_reactions(
    self, inputs: List[Molecule], num_results: int
) -> List[Sequence[SingleProductReaction]]:
    """Generate reaction predictions for input molecules."""
    # 具体实现代码...
```

### 7.3 SimpRetroModel._get_reactions 详细讲解

这个方法实现了**模板匹配 + 神经网络过滤**的逆合成预测策略。

#### 整体流程图：

```
输入：分子列表
    ↓
┌─────────────────────────────────────────┐
│  对每个分子：                            │
│                                         │
│  阶段1：模板匹配 (Phase 1)              │
│  ┌─────────────────────────────────┐   │
│  │ 遍历所有反应模板                 │   │
│  │   ↓                             │   │
│  │ 用 rdchiral 尝试应用模板        │   │
│  │   ↓                             │   │
│  │ 如果匹配成功：                   │   │
│  │   - 计算各种得分                 │   │
│  │   - 存入 results 字典            │   │
│  └─────────────────────────────────┘   │
│           ↓                             │
│  阶段2：神经网络过滤 (Phase 2)          │
│  ┌─────────────────────────────────┐   │
│  │ 将分子和模板转换为指纹           │   │
│  │   ↓                             │   │
│  │ 用神经网络评估每个反应可能性     │   │
│  │   ↓                             │   │
│  │ 过滤掉低置信度的反应             │   │
│  └─────────────────────────────────┘   │
│           ↓                             │
│  阶段3：排序和选择 (Phase 3)            │
│  ┌─────────────────────────────────┐   │
│  │ 按得分排序                       │   │
│  │   ↓                             │   │
│  │ 选择前 num_results 个结果        │   │
│  │   ↓                             │   │
│  │ 转换为概率分布 (softmax)         │   │
│  └─────────────────────────────────┘   │
│           ↓                             │
│  转换为 Reaction 对象                   │
└─────────────────────────────────────────┘
    ↓
输出：反应列表
```

#### 逐段代码详解：

**第205-207行：初始化变量**
```python
raw_outputs = []  # 存储所有输出
w1, w2, w3, w4 = 0.1, 0.2, 0.5, 0  # 四个评分权重
threshold = 0.2  # 神经网络过滤阈值
```

**权重含义：**
- `w1 = 0.1`：复杂度差异得分 (CDScore) 权重
- `w2 = 0.2`：可用性得分 (ASScore) 权重
- `w3 = 0.5`：环差异得分 (RDScore) 权重
- `w4 = 0`：模板匹配数量的权重（设为0表示不使用）

**第209-237行：阶段1 - 模板匹配**
```python
for x in inputs:  # 遍历每个输入分子
    results = {}  # 存储结果：{产物SMILES: (得分, 模板, ID, 环得分)}
    result_set = set([])  # 用于去重
    p_mol = Chem.MolFromSmiles(x.smiles)  # RDKit分子对象
    p_mol_rdchiral = rdchiralReactants(x.smiles)  # rdchiral反应物对象
    valid_template_id = []  # 记录有效的模板ID

    # 对每个模板进行匹配
    for idx, (template, template_raw) in enumerate(
        zip(self.template_list, self.templates_raw)
    ):
        # 尝试用模板匹配产物
        mapped_curr_results = rdchiralRun(template, p_mol_rdchiral, keep_mapnums=True)

        # 处理每个匹配结果
        for r in mapped_curr_results:
            canonical_r = canonical_smiles(r)  # 转为标准SMILES
            canonical_r_dict = {r_: canonical_smiles(r_) for r_ in r.split(".")}
            if idx not in valid_template_id:
                valid_template_id.append(idx)
            if canonical_r in result_set:
                continue  # 跳过重复
            result_set.add(canonical_r)
            r_mols = [Chem.MolFromSmiles(r_) for r_ in r.split(".")]

            # 计算各种得分
            rdscore = RDScore(p_mol, r_mols)  # 环差异得分
            score = 1 * (  # 综合得分
                w1 * CDScore(p_mol, r.split(".")) +
                w2 * ASScore(p_mol, canonical_r_dict, self.instock_list) +
                w3 * rdscore +
                w4 * 1 / len(mapped_curr_results)
            )
            results[canonical_r] = (score, template_raw, idx, rdscore)
```

**什么是模板匹配？**

反应模板就像"菜谱"，描述了如何从产物倒推原料。例如：

```
模板：[C:1]=[O:2][C:3] >> [C:1][O:2].[C:3]
含义：羰基化合物的C-C键断裂，生成两个片段
```

**第239-262行：阶段2 - 神经网络过滤**
```python
# 准备神经网络输入
valid_temp_fps = self.template_fps[valid_template_id]  # 有效模板的指纹
p_fp = smiles_to_fingerprint(x.smiles)  # 产物分子的指纹

# 构建神经网络输入数据
data = torch.tensor(
    np.concatenate(
        [valid_temp_fps.squeeze(), np.repeat(p_fp, len(valid_temp_fps), axis=0)],
        axis=1,
    ),
    dtype=torch.float32,
)

# 神经网络预测
with torch.no_grad():
    pred = self.filter(data).squeeze().cpu().numpy()

# 过滤低置信度的反应
validated_results = {}
for i, (k, v) in enumerate(results.items()):
    if pred[valid_template_id.index(v[2])] > threshold or v[-1]:
        validated_results[k] = (v[0], v[1], v[2], pred[...])
```

**为什么需要神经网络过滤？**
- 模板匹配可能产生很多无效或不合理的反应
- 神经网络经过训练，可以判断反应是否"合理"
- 阈值 0.2 表示只有置信度 > 20% 的反应才会被保留

**第264-284行：阶段3 - 排序、选择和概率化**
```python
# 按得分排序，选择前 num_results 个
results = sorted(
    validated_results.items(),
    key=lambda item: item[1][0] + 0.001 * item[1][-1],  # 综合得分
    reverse=True,  # 降序
)[:num_results]

if len(results) > 0:
    reactants, scores = zip(*results)
    templates = [t[1] for t in scores]
    scores = [s[0] for s in scores]

    # Softmax：将得分转换为概率分布
    scores = [np.exp(s) for s in scores]
    total = sum(scores)
    if total > 0:
        scores = [s / total for s in scores]
    else:
        scores = [1.0 / len(scores)] * len(scores)

    raw_outputs.append((reactants, scores, templates))
else:
    raw_outputs.append(([], [], []))
```

**什么是 Softmax？**

Softmax 是一种数学变换，将任意数值转换为概率分布：

```python
原始得分: [2.5, 1.0, 0.5]
    ↓ e^x
中间值: [12.18, 2.72, 1.65]
    ↓ 归一化
最终概率: [0.74, 0.16, 0.10]  # 总和为1
```

**第286-294行：转换为标准格式**
```python
return [
    process_raw_smiles_outputs_backwards(
        input=input,
        output_list=output[0],  # 反应物 SMILES
        metadata_list=[  # 元数据（概率和模板）
            {"probability": score, "template": temp_smarts}
            for score, temp_smarts in zip(output[1], output[2])
        ],
    )
    for input, output in zip(inputs, raw_outputs)
]
```

这一步将原始输出转换为标准的 `SingleProductReaction` 对象。

### 7.4 完整示例

假设我们要预测分子 `CC(=O)O`（乙酸）的逆合成反应：

```python
# 输入
inputs = [Molecule("CC(=O)O")]
num_results = 10

# 调用 _get_reactions
results = model._get_reactions(inputs, num_results)

# 内部流程：
# 1. 模板匹配：
#    - 尝试数千个反应模板
#    - 找到50个匹配的反应
#
# 2. 神经网络过滤：
#    - 计算每个反应的置信度
#    - 过滤后保留20个反应
#
# 3. 排序和选择：
#    - 按综合得分排序
#    - 选择前10个
#    - 转换为概率分布
#
# 输出：
# [
#   [
#     Reaction(reactants=["CO", "C=O"], probability=0.35, template="..."),
#     Reaction(reactants=["CC(=O)O"], probability=0.25, template="..."),
#     ...
#   ]
# ]
```

### 7.5 得分计算详解

**CDScore (Complexity Difference Score) - 复杂度差异：**
```python
def CDScore(p_mol, r_mols):
    """比较产物和反应物的原子数"""
    p_atom_count = p_mol.GetNumAtoms()
    r_atom_count = [计算每个反应物的原子数]
    ...
    return 1 / (1 + MAE) * p_atom_count
```

**ASScore (Availability Score) - 可用性得分：**
```python
def ASScore(p_mol, r_mol_dict, in_stock):
    """反应物是否在库存中"""
    for k, v in r_mol_dict.items():
        if v in in_stock:
            asscore += ...
    return asscore
```

**RDScore (Ring Difference Score) - 环差异得分：**
```python
def RDScore(p_mol, r_mols):
    """比较产物和反应物的环数量"""
    p_ring_count = p_mol.GetRingInfo().NumRings()
    r_ring_count = 计算反应物的环数量
    if p_ring_count > r_ring_count:
        return 1  # 开环反应，加分
    else:
        return 0
```

---

## 八、总结

### 8.1 设计模式

```
┌─────────────────────────────────────────────────────┐
│              ReactionModel (基类)                    │
│  • 定义接口                                          │
│  • 实现缓存逻辑                                      │
│  • _get_reactions 是抽象方法                         │
└─────────────────────────────────────────────────────┘
                        ↑
                        | 继承
                        |
┌─────────────────────────────────────────────────────┐
│          SimpRetroModel (具体实现)                  │
│  • 实现了 _get_reactions 方法                        │
│  • 使用模板匹配 + 神经网络过滤                        │
└─────────────────────────────────────────────────────┘
```

### 8.2 关键要点

1. **抽象方法**：`_get_reactions` 在基类中没有实现，子类必须实现
2. **缓存机制**：`__call__` 方法自动处理缓存，避免重复计算
3. **多态性**：不同的模型类可以有不同的 `_get_reactions` 实现
4. **组合模式**：模板匹配 + 神经网络 = 更好的预测效果

### 8.3 学习路径建议

作为高中生，建议按以下顺序深入理解：

1. ✅ 理解基类 `ReactionModel` 的结构
2. ✅ 理解 `__call__` 方法的缓存逻辑
3. ✅ 理解 `_get_reactions` 的作用
4. ✅ 查看具体实现（如 `SimpRetroModel`）
5. ✅ 理解模板匹配和神经网络过滤
6. ⬜ 学习 RDKit 化学信息学基础
7. ⬜ 学习 PyTorch 神经网络基础
8. ⬜ 阅读其他模型实现（如 Graph-based 模型）

---

## 九、文件引用索引

| 文件 | 路径 | 说明 |
|------|------|------|
| models.py | [interface/models.py](syntheseus/interface/models.py) | 基类定义 |
| simpretro.py | [reaction_prediction/inference/simpretro.py](syntheseus/reaction_prediction/inference/simpretro.py) | SimpRetro 实现 |
| inference_base.py | [reaction_prediction/inference_base.py](syntheseus/reaction_prediction/inference_base.py) | ExternalBackwardReactionModel |
