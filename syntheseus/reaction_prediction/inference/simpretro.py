"""Inference wrapper for the SimpRetro model.

Template-based retrosynthesis model with neural network filtering.
Combines template matching with heuristics and a fast neural filter.
"""

import importlib.resources
import json
import pathlib
import pickle
import re
from pathlib import Path
from typing import List, Sequence, Union

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
from rdchiral.initialization import rdchiralReaction, rdchiralReactants
from rdchiral.main import rdchiralRun
from tqdm import tqdm

from syntheseus.interface.molecule import Molecule
from syntheseus.interface.reaction import SingleProductReaction
from syntheseus.reaction_prediction.inference_base import ExternalBackwardReactionModel
from syntheseus.reaction_prediction.utils.inference import process_raw_smiles_outputs_backwards
from syntheseus.reaction_prediction.fast_filter.model import (
    Net_orig,
    fingerprint_base as filter_fingerprint_base,
    save_fingerprint_base,
)


def CDScore(p_mol, r_mols):
    """Calculate complexity difference score between product and reactants."""
    p_atom_count = p_mol.GetNumAtoms()
    n_r_mols = len(r_mols)
    if n_r_mols == 1:
        return 0
    r_atom_count = [
        len([int(num[1:]) for num in re.findall(r":\d+", r_mol) if int(num[1:]) < 900])
        for r_mol in r_mols
    ]
    main_r = r_mols[np.argmax(r_atom_count)]
    if len(Chem.MolFromSmiles(main_r).GetAtoms()) >= p_atom_count:
        return 0
    MAE = 1 / n_r_mols * sum(
        [abs(p_atom_count / n_r_mols - r_atom_count[i]) for i in range(n_r_mols)]
    )
    return 1 / (1 + MAE) * p_atom_count


def ASScore(p_mol, r_mol_dict, in_stock):
    """Calculate availability score for reactants."""
    p_atom_count = p_mol.GetNumAtoms()
    r_mols = list(r_mol_dict.keys())
    r_atom_count = [
        len([int(num[1:]) for num in re.findall(r":\d+", r_mol) if int(num[1:]) < 900])
        for r_mol in r_mols
    ]
    main_r = r_mols[np.argmax(r_atom_count)]
    asscore = 0
    for k, v in r_mol_dict.items():
        if v in in_stock:
            add = len(
                [int(num[1:]) for num in re.findall(r":\d+", k) if int(num[1:]) < 900]
            )
            if len(Chem.MolFromSmiles(main_r).GetAtoms()) < p_atom_count:
                asscore += add
            else:
                asscore += add if add > 2 else 0
        if ("Mg" in v or "Li" in v or "Zn" in v) and v not in in_stock:
            asscore -= 10
    return asscore


def RDScore(p_mol, r_mols):
    """Calculate ring difference score."""
    p_ring_count = p_mol.GetRingInfo().NumRings()
    r_rings_s = [r_mol.GetRingInfo().AtomRings() for r_mol in r_mols]
    r_ring_count = 0
    for r_rings, r_mol in zip(r_rings_s, r_mols):
        for r_ring in r_rings:
            mapnums = [r_mol.GetAtomWithIdx(i).GetAtomMapNum() for i in r_ring]
            symbols = [r_mol.GetAtomWithIdx(i).GetSymbol() for i in r_ring]
            if "B" in symbols or "Si" in symbols:
                continue
            if min(mapnums) < 900:
                r_ring_count += 1
    if p_ring_count > r_ring_count:
        return 1
    else:
        return 0


def canonical_smiles(smiles):
    """Convert SMILES to canonical form without atom mapping."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(mol)))


def smiles_to_fingerprint(smiles, fp_length=2048, radius=2):
    """Generate Morgan fingerprint from SMILES using the new RDKit API."""
    mol = Chem.MolFromSmiles(smiles)
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_length)
    fp = fpgen.GetFingerprint(mol)
    return np.array(fp).reshape(1, -1)


def smarts_to_fingerprint(smarts):
    """Generate fingerprint from reaction SMARTS template."""
    rxn = AllChem.ReactionFromSmarts(smarts)
    return np.concatenate(
        [
            np.array(AllChem.CreateDifferenceFingerprintForReaction(rxn).ToList()).reshape(1, -1),
            np.array(AllChem.CreateStructuralFingerprintForReaction(rxn).ToList()).reshape(1, -1),
        ],
        axis=1,
    )


class SimpRetroModel(ExternalBackwardReactionModel):
    """Template-based retrosynthesis model with neural network filtering."""

    def __init__(
        self,
        model_dir: Union[str, Path],
        device: str,
        inventory_file: Union[str, Path] = "emolecules.txt",
        **kwargs
    ) -> None:
        """Initialize SimpRetro model.

        Args:
            model_dir: Path to template JSON file
            device: Device for neural filter ('cpu' or 'cuda')
            inventory_file: Path to inventory/building block file (default: emolecules.txt)
        """
        super().__init__(model_dir=model_dir, device=device, **kwargs)

        # Load reaction templates
        self.templates_raw = json.load(open(self.model_dir))
        print(f"Total Number of Templates: {len(self.templates_raw)}")
        self.template_list = []
        for i, l in tqdm(enumerate(self.templates_raw), desc="loading templates"):
            rule = l.strip()
            self.template_list.append(rdchiralReaction(rule))

        # Use shared fingerprint_base from fast_filter module
        self.fingerprint_base = filter_fingerprint_base
        self.template_fps = []
        for template in self.templates_raw:
            if template in self.fingerprint_base:
                self.template_fps.append(self.fingerprint_base[template])
            else:
                fp = smarts_to_fingerprint(template)
                self.template_fps.append(fp)
                self.fingerprint_base[template] = fp
        self.template_fps = np.array(self.template_fps)
        # Save updated fingerprints
        save_fingerprint_base()

        # Load in-stock molecule list
        inventory_path = Path(inventory_file)
        if not inventory_path.is_absolute():
            # If relative path, try current directory first, then model directory
            if inventory_path.exists():
                inventory_path = inventory_path.resolve()
            else:
                inventory_path = Path(self.model_dir).parent / inventory_file
        self.instock_list = set(open(inventory_path).read().split("\n"))
        print(f"Number of in-stock molecules: {len(self.instock_list)}")

        # Load neural network filter
        self.filter = Net_orig()
        # Load model weights from package data
        with importlib.resources.files(
            "syntheseus.reaction_prediction.fast_filter"
        ).joinpath("model_smoothbce.pth") as model_path:
            self.filter.load_state_dict(
                torch.load(str(model_path), map_location=self.device)
            )

    def get_parameters(self):
        """Return model parameters for optimization."""
        return self.filter.parameters()

    def _get_reactions(
        self, inputs: List[Molecule], num_results: int
    ) -> List[Sequence[SingleProductReaction]]:
        """Generate reaction predictions for input molecules.

        Args:
            inputs: List of product molecules
            num_results: Maximum number of predictions per molecule

        Returns:
            List of reaction predictions for each input molecule
        """
        raw_outputs = []
        w1, w2, w3, w4 = 0.1, 0.2, 0.5, 0
        threshold = 0.2

        for x in inputs:
            results = {}
            result_set = set([])
            p_mol = Chem.MolFromSmiles(x.smiles)
            p_mol_rdchiral = rdchiralReactants(x.smiles)
            valid_template_id = []

            # Template matching phase
            for idx, (template, template_raw) in enumerate(
                zip(self.template_list, self.templates_raw)
            ):
                mapped_curr_results = rdchiralRun(template, p_mol_rdchiral, keep_mapnums=True)
                for r in mapped_curr_results:
                    canonical_r = canonical_smiles(r)
                    canonical_r_dict = {r_: canonical_smiles(r_) for r_ in r.split(".")}
                    if idx not in valid_template_id:
                        valid_template_id.append(idx)
                    if canonical_r in result_set:
                        continue
                    result_set.add(canonical_r)
                    r_mols = [Chem.MolFromSmiles(r_) for r_ in r.split(".")]
                    rdscore = RDScore(p_mol, r_mols)
                    score = 1 * (
                        w1 * CDScore(p_mol, r.split("."))
                        + w2 * ASScore(p_mol, canonical_r_dict, self.instock_list)
                        + w3 * rdscore
                        + w4 * 1 / len(mapped_curr_results)
                    )
                    results[canonical_r] = (score, template_raw, idx, rdscore)

            # Neural network filtering phase
            valid_temp_fps = self.template_fps[valid_template_id]
            p_fp = smiles_to_fingerprint(x.smiles)
            try:
                data = torch.tensor(
                    np.concatenate(
                        [valid_temp_fps.squeeze(), np.repeat(p_fp, len(valid_temp_fps), axis=0)],
                        axis=1,
                    ),
                    dtype=torch.float32,
                )
                with torch.no_grad():
                    pred = self.filter(data).squeeze().cpu().numpy()
                validated_results = {}
                for i, (k, v) in enumerate(results.items()):
                    if pred[valid_template_id.index(v[2])] > threshold or v[-1]:
                        validated_results[k] = (
                            v[0],
                            v[1],
                            v[2],
                            pred[valid_template_id.index(v[2])],
                        )
            except Exception:
                validated_results = {}

            # Sort and select top results
            results = sorted(
                validated_results.items(),
                key=lambda item: item[1][0] + 0.001 * item[1][-1],
                reverse=True,
            )[:num_results]

            if len(results) > 0:
                reactants, scores = zip(*results)
                templates = [t[1] for t in scores]
                scores = [s[0] for s in scores]
                # Convert scores to probabilities using softmax
                scores = [np.exp(s) for s in scores]
                total = sum(scores)
                if total > 0:
                    scores = [s / total for s in scores]
                else:
                    scores = [1.0 / len(scores)] * len(scores)
                raw_outputs.append((reactants, scores))
            else:
                raw_outputs.append(([], []))

        # Convert to new format using process_raw_smiles_outputs_backwards
        return [
            process_raw_smiles_outputs_backwards(
                input=input,
                output_list=output[0],
                metadata_list=[{"probability": score} for score in output[1]],
            )
            for input, output in zip(inputs, raw_outputs)
        ]
