"""Fast filter neural network for SimpRetro.

This module provides a neural network that filters reaction predictions
based on template and product fingerprints.
"""

import pathlib
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator

# Use cache directory for fingerprint storage
cache_dir = pathlib.Path.home() / ".cache" / "simpretro"
cache_dir.mkdir(parents=True, exist_ok=True)
fingerprint_cache = cache_dir / "fingerprint_base.pkl"

if fingerprint_cache.exists():
    fingerprint_base = pickle.load(open(fingerprint_cache, "rb"))
else:
    fingerprint_base = {}


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
            np.array(AllChem.CreateDifferenceFingerprintForReaction(rxn).ToList()).reshape(
                1, -1
            ),
            np.array(AllChem.CreateStructuralFingerprintForReaction(rxn).ToList()).reshape(
                1, -1
            ),
        ],
        axis=1,
    )


def save_fingerprint_base():
    """Save fingerprint base to cache."""
    pickle.dump(fingerprint_base, open(fingerprint_cache, "wb"))


def get_fps_for_temps(temps):
    """Get fingerprints for reaction templates.

    Args:
        temps: List of reaction SMARTS templates

    Returns:
        numpy array of fingerprints
    """
    fps = []
    length = len(fingerprint_base)
    for temp in temps:
        if temp not in fingerprint_base:
            fingerprint_base[temp] = smarts_to_fingerprint(temp)
        fps.append(fingerprint_base[temp])
    if len(fingerprint_base) > length:
        save_fingerprint_base()
    return np.array(fps)


class Net_orig(nn.Module):
    """Neural network for filtering reaction predictions."""

    def __init__(self):
        super(Net_orig, self).__init__()
        self.fc1 = nn.Linear(2048 * 4, 2048)
        self.fc2 = nn.Linear(2048, 1)
        self.bn = nn.BatchNorm1d(2048)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.bn(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x
