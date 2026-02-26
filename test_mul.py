from syntheseus import Molecule
from syntheseus.reaction_prediction.inference import LocalRetroModel, RetroKNNModel, RootAlignedModel

test_mol = Molecule("Cc1ccc(-c2ccc(C)cc2)cc1")
# model = LocalRetroModel()
# model = RetroKNNModel()
# model = RootAlignedModel()

# print(model.model_dir)

# def mols_to_str(mols) -> str:
#     return " + ".join([mol.smiles for mol in mols])

# def print_results(results) -> None:
#     for idx, prediction in enumerate(results):
#         print(f"{idx + 1}: " + mols_to_str(prediction.reactants))

# [results] = model([test_mol], num_results=5)
# print_results(results)
from syntheseus.search.mol_inventory import SmilesListInventory
from syntheseus.search.algorithms.breadth_first import (
    AndOr_BreadthFirstSearch
)

# Set up a reaction model with caching enabled. Number of reactions
# to request from the model at each step of the search needs to be
# provided at construction time.
model = LocalRetroModel(use_cache=True, default_num_results=10)

from syntheseus_paroutes import PaRoutesInventory
from pprint import pprint
inventory = PaRoutesInventory(n=5)
print(f"Size of inventory: {len(inventory)}")
print("First few molecules in inventory:")
pprint(list(inventory.to_purchasable_mols())[:5])
# Dummy inventory with just two purchasable molecules.
# inventory = SmilesListInventory(
#     smiles_list=["Cc1ccc(B(O)O)cc1", "O=Cc1ccc(I)cc1"]
# )
# from syntheseus import MoleculeInventory
# inventory = MoleculeInventory


search_algorithm = AndOr_BreadthFirstSearch(
    reaction_model=model,
    mol_inventory=inventory,
    limit_iterations=100,  # max number of algorithm iterations
    limit_reaction_model_calls=100,  # max number of model calls
    time_limit_s=60.0  # max runtime in seconds
)

output_graph, _ = search_algorithm.run_from_mol(test_mol)

print(f"Explored {len(output_graph)} nodes")

from syntheseus.search.analysis.route_extraction import (
    iter_routes_time_order,
)
from syntheseus.search.graph.and_or import AndNode

# Extract the routes simply in the order they were found.
routes = list(iter_routes_time_order(output_graph, max_routes=10))

for idx, route in enumerate(routes):
    num_reactions = len({n for n in route if isinstance(n, AndNode)})
    print(f"Route {idx + 1} consists of {num_reactions} reactions")
    # print(route)

from syntheseus.search.visualization import visualize_andor

for idx, route in enumerate(routes):
    visualize_andor(
        output_graph, filename=f"route_p_{idx + 1}.pdf", nodes=route
    )




