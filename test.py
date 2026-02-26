from syntheseus import Molecule
from syntheseus.reaction_prediction.inference import LocalRetroModel, RetroKNNModel, SimpRetro

test_mol = Molecule("Cc1ccc(-c2ccc(C)cc2)cc1")
# model = LocalRetroModel()
# model = RetroKNNModel()
model = SimpRetro()

print(model.model_dir)

def mols_to_str(mols) -> str:
    return " + ".join([mol.smiles for mol in mols])

def print_results(results) -> None:
    for idx, prediction in enumerate(results):
        print(f"{idx + 1}: " + mols_to_str(prediction.reactants))

[results] = model([test_mol], num_results=5)
print_results(results)