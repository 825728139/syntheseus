from typing import Dict, Any, List

from syntheseus import Molecule
from syntheseus.search.mol_inventory import SmilesListInventory
from syntheseus.reaction_prediction.inference import LocalRetroModel
# 注意：你需要确保已经下载并配置好了 LocalRetro 的权重，或者换成其他的模型
# 如果没有权重，这里只是代码示例，无法真正运行出结果

from syntheseus.search.algorithms.breadth_first import AndOr_BreadthFirstSearch
from syntheseus.search.analysis.route_extraction import iter_routes_time_order
from syntheseus.search.graph.and_or import AndNode
from syntheseus.search.visualization import visualize_andor

# ==========================================
# 1. 自定义带价格的库存类
# ==========================================
class PricedDictInventory(SmilesListInventory):
    """
    一个简单的基于字典的库存，可以存储分子的元数据（价格、链接等）。
    """
    def __init__(self, purchasable_db: Dict[str, Dict[str, Any]]):
        """
        Args:
            purchasable_db: 字典，格式为 { "SMILES_STRING": { "price": ..., "url": ... } }
        """
        # 为了快速查找，我们将输入的 SMILES 进行标准化（Canonicalize）作为 key
        self._inventory = {}
        for smi, metadata in purchasable_db.items():
            try:
                # 尝试标准化 SMILES，保证匹配准确率
                mol = Molecule(smi) # Syntheseus 的 Molecule 会自动处理标准化
                self._inventory[mol.smiles] = metadata
            except Exception:
                pass # 忽略无效 SMILES

    def is_purchasable(self, mol: Molecule) -> bool:
        """核心接口：判断分子是否可购买，并注入 Metadata"""
        if mol.smiles in self._inventory:
            # 【关键步骤】将库存中的信息注入到分子的 metadata 中
            # 这样在后续提取路线时，这些信息依然存在
            data = self._inventory[mol.smiles]
            mol.metadata.update(data)
            mol.metadata["is_purchasable"] = True
            return True
        return False

# ==========================================
# 2. 准备测试数据 (Inventory Variable)
# ==========================================

# 这是一个模拟的“数据库”，包含价格、包装、链接
# 这里的分子是 4,4'-Dimethylbiphenyl (目标分子) 可能的合成原料
# 反应路径可能是：Suzuki Coupling
# 原料 A: 4-Tolylboronic acid (Cc1ccc(B(O)O)cc1)
# 原料 B: 4-Iodotoluene (Cc1ccc(I)cc1) 或 4-Bromotoluene
inventory_data = {
    "Cc1ccc(B(O)O)cc1": {
        "price_per_g": 25.0,
        "currency": "USD",
        "pack_size": "5g",
        "vendor": "Sigma-Aldrich",
        "url": "https://www.sigmaaldrich.com/US/en/product/aldrich/393622"
    },
    "Cc1ccc(I)cc1": {
        "price_per_g": 18.5,
        "currency": "USD",
        "pack_size": "25g",
        "vendor": "Enamine",
        "url": "https://store.enamine.net/catalog/product/view/id/12345"
    },
    # 也可以加一些常用的溶剂或简单试剂，防止搜索因为缺简单原料而失败
    "CC(=O)O": { "price_per_g": 0.1, "pack_size": "1L", "vendor": "Generic" }, # 乙酸
    "CCO": { "price_per_g": 0.05, "pack_size": "1L", "vendor": "Generic" },    # 乙醇
}

# 实例化库存对象
inventory = PricedDictInventory(inventory_data)

# ==========================================
# 3. 运行搜索 (配合你的原始逻辑)
# ==========================================

test_mol = Molecule("Cc1ccc(-c2ccc(C)cc2)cc1") # 目标：4,4'-Dimethylbiphenyl

# 注意：这里需要你有真实可用的模型权重。
# 如果没有，代码会报错。这里假设你已经搞定了 LocalRetroModel
try:
    model = LocalRetroModel(use_cache=True, default_num_results=10)

    search_algorithm = AndOr_BreadthFirstSearch(
        reaction_model=model,
        mol_inventory=inventory, # 使用我们要带价格的 inventory
        limit_iterations=20,     # 稍微改小一点以便快速测试
        limit_reaction_model_calls=50,
        time_limit_s=60.0
    )

    print("开始搜索...")
    output_graph, _ = search_algorithm.run_from_mol(test_mol)
    print(f"搜索结束，探索了 {len(output_graph)} 个节点")

    # ==========================================
    # 4. 验证结果：提取路线并打印价格信息
    # ==========================================
    routes = list(iter_routes_time_order(output_graph, max_routes=5))

    if not routes:
        print("未找到合成路线。可能原因：模型未加载、库存不足或搜索步数太少。")
    else:
        for idx, route in enumerate(routes):
            print(f"\n--- 路线 {idx + 1} ---")
            
            # 遍历路线中的所有节点
            # 路线是一个 set of nodes，我们需要找出其中的起始原料（叶子节点）
            starting_materials = []
            urls = []
            total_cost = 0.0
            
            for node in route:
                # 在 Syntheseus Graph 中，OrNode 代表分子
                # 如果它没有子节点（children），或者在 inventory 中，就是起始原料
                # 注意：route 对象是 Set[Node]，直接判断 metadata 最准
                if hasattr(node, "mol"): # 确保是分子节点
                    mol = node.mol
                    if mol.metadata.get("is_purchasable"):
                        price = mol.metadata.get("price_per_g", 0)
                        vendor = mol.metadata.get("vendor", "Unknown")
                        starting_materials.append(f"{mol.smiles} ({vendor}, ${price}/g)")
                        urls.append(mol.metadata.get("url"))
                        total_cost += price
            
            print("起始原料:")
            for sm in starting_materials:
                print(f"  - {sm}")
            print(urls)
            print(f"预估原料总单价: ${total_cost:.2f}")

except Exception as e:
    print(f"运行出错 (可能是模型权重路径问题): {e}")
    print("提示：请确保 LocalRetroModel 能正确加载，或者替换为其他可用模型。")
