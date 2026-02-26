import json                                                                                                                                                                                             
import datetime                                                                                                                                                                                         
                                                                                                                                                                                                        
def make_serializable(obj):                                                                                                                                                                             
    """递归将不可序列化的对象转为字符串"""                                                                                                                                                              
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, datetime.datetime):
        return obj.isoformat()
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    else:
        return f"{str(obj)}"  # Mol 对象等其他类型转为字符串

def serialize_node(node):
    base = {
        'id': str(id(node)),
        'has_solution': node.has_solution,
        'num_visit': node.num_visit,
        'depth': node.depth,
        'is_expanded': node.is_expanded,
        'creation_time': node.creation_time.isoformat() if node.creation_time else None,
        'data': make_serializable(dict(node.data)),
    }

    if hasattr(node, 'mol'):  # OrNode (retro_star算法使用)
        base.update({
            'type': 'OrNode',
            'smiles': node.mol.smiles,
            'is_purchasable': node.mol.metadata.get('is_purchasable', False),
            'mol_metadata': make_serializable(dict(node.mol.metadata)),
        })
    elif hasattr(node, 'reaction'):  # AndNode (retro_star算法使用)
        base.update({
            'type': 'AndNode',
            'reaction_smiles': node.reaction.reaction_smiles,
            'reactants': [r.smiles for r in node.reaction.reactants],
            'product': node.reaction.product.smiles,
            'reaction_metadata': make_serializable(dict(node.reaction.metadata)),
        })
    elif hasattr(node, 'mols'):  # MolSetNode (MCTS算法使用)
        base.update({
            'type': 'MolSetNode',
            'smiles_list': sorted([mol.smiles for mol in node.mols]),
            'is_purchasable': all(mol.metadata.get('is_purchasable', False) for mol in node.mols),
            'mols_metadata': [make_serializable(dict(mol.metadata)) for mol in node.mols],
        })
    return base


def serialize_reaction(reaction):
    """序列化反应对象，包含元数据（probability、log_probability、confidence、score等）"""
    return {
        'reaction_smiles': reaction.reaction_smiles,
        'reactants': [r.smiles for r in reaction.reactants],
        'product': reaction.product.smiles,
        'metadata': make_serializable(dict(reaction.metadata)),
    }


if __name__=="__main__":
    output_graph={}
    # 序列化并保存，兼容两种图类型（retro_star和MCTS）
    node_to_id = {node: i for i, node in enumerate(output_graph._graph.nodes)}

    # 获取根节点的SMILES，兼容两种图类型
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

    with open('/home/liwenlong/graph_output.json', 'w') as f:
        json.dump(output, f, indent=2)