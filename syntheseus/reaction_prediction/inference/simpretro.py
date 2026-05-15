"""Inference wrapper for the SimpRetro model.

Template-based retrosynthesis model with neural network filtering.
Combines template matching with heuristics and a fast neural filter.
"""

# Must be set BEFORE importing PyTorch / numpy to prevent fork+OpenMP deadlock.
import atexit
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import importlib.resources
import json
import multiprocessing as mp
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator

from syntheseus.interface.molecule import Molecule
from syntheseus.interface.reaction import SingleProductReaction
from syntheseus.reaction_prediction.inference_base import ExternalBackwardReactionModel
from syntheseus.reaction_prediction.utils.inference import process_raw_smiles_outputs_backwards
from syntheseus.reaction_prediction.fast_filter.model import (
    Net_orig,
    fingerprint_base as filter_fingerprint_base,
    save_fingerprint_base,
)
from syntheseus.reaction_prediction.inference.simpretro_match import (
    TemplateLibrary,
    match_all_templates,
    MatchConfig,
)


# ---------------------------------------------------------------------------
# Sub-process worker: runs the full pipeline for a single molecule.
# Fork COW: children inherit _MATCH_MODEL set by __init__ (no pickling needed).
# ---------------------------------------------------------------------------
_MATCH_MODEL: Optional["SimpRetroModel"] = None


def _match_single_molecule(smiles: str, num_results: int, threshold: float = 0.2):
    """Full pipeline inside forked subprocess: matching + neural filter + sort + probability."""
    # Phase 1: C++ template matching
    out = match_all_templates(
        product_smiles=smiles,
        template_library=_MATCH_MODEL.cpp_template_lib,
        config=_MATCH_MODEL.match_config,
    )
    # Reproduce old version data structures:
    # results = {canonical_r: (score, template_raw, idx, rdscore)}
    # valid_template_id = [idx, ...]
    results = dict(out.results)
    valid_template_id = list(out.valid_template_ids)

    # --- Neural network filtering: copied verbatim from simpretro_old.py:244-267 ---
    valid_temp_fps = _MATCH_MODEL.template_fps[valid_template_id]
    p_fp = smiles_to_fingerprint(smiles)
    try:
        data = torch.tensor(
            np.concatenate(
                [valid_temp_fps.squeeze(), np.repeat(p_fp, len(valid_temp_fps), axis=0)],
                axis=1,
            ),
            dtype=torch.float32,
        )
        with torch.no_grad():
            pred = _MATCH_MODEL.filter(data).squeeze().cpu().numpy()
        validated_results = {}
        for i, (k, v) in enumerate(results.items()):
            if pred[valid_template_id.index(v[2])] > threshold or v[-1]:
                validated_results[k] = (
                    v[0],
                    v[1],
                    v[2],
                    pred[valid_template_id.index(v[2])],
                )
    except Exception as e:
        print(f"Error in neural filter: {e}")
        validated_results = {}

    # --- Sort and select top results: copied verbatim from simpretro_old.py:270-289 ---
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
        probability = [np.exp(s) for s in scores]
        total = sum(probability)
        if total > 0:
            probability = [p / total for p in probability]
        else:
            probability = [1.0 / len(probability)] * len(probability)
        return (reactants, probability, scores, templates)
    else:
        return ([], [], [], [])


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
        inventory_file: Union[str, Path, set] = "/home/liwenlong/chemTools/retro_syn/syntheseus/emolecules.txt",
        **kwargs
    ) -> None:
        """Initialize SimpRetro model.

        Args:
            model_dir: Path to template JSON file
            device: Device for neural filter ('cpu' or 'cuda')
            inventory_file: Path to inventory file, or a pre-loaded set of SMILES.
                If a set is passed, it is written to a temp file for C++ loading.
        """
        super().__init__(model_dir=model_dir, device=device, **kwargs)

        # Load reaction templates
        self.templates_raw = json.load(open(self.model_dir))
        print(f"Total Number of Templates: {len(self.templates_raw)}")

        # C++ TemplateLibrary (pre-parsed, fast matching)
        print("Building C++ TemplateLibrary ...")
        self.cpp_template_lib = TemplateLibrary(self.templates_raw)

        # Handle inventory: file path (C++ reads directly) or set (write to temp for C++)
        if isinstance(inventory_file, set):
            import tempfile
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", prefix="simpretro_inv_", delete=False
            )
            for smi in inventory_file:
                tmp.write(smi + "\n")
            tmp.close()
            self.cpp_template_lib.set_inventory_file(tmp.name)
            print(f"Wrote {len(inventory_file)} inventory SMILES to temp file: {tmp.name}")
        else:
            self.cpp_template_lib.set_inventory_file(str(inventory_file))

        # C++ reads private templates directly from JSON file
        self.cpp_template_lib.set_private_templates_file(
            str(Path("/home/liwenlong/chemTools/retro_syn/syntheseus/syntheseus/cli/private_templates.json"))
        )

        # Build MatchConfig — heuristic scoring weights
        self.match_config = MatchConfig()
        self.match_config.w_cd = 0.1       # CDScore: 复杂度差异得分，反应物中参与反应的原子数量
        self.match_config.w_as = 0.2       # ASScore: 可用性得分，依赖库存中的反应物参与反应的原子数量
        self.match_config.w_rd = 0.5       # RDScore: 环形差异得分，返回值0或1（产物比反应物多环则+1）
        self.match_config.w_md = 0.0       # MDScore: 多样性得分，预测反应物个数的倒数
        self.match_config.w_private = 0.0  # private_bonus: 私有模板加成权重（private_bonus=10.0）

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

        # Load neural network filter
        self.filter = Net_orig()
        # Load model weights from package data
        with importlib.resources.files(
            "syntheseus.reaction_prediction.fast_filter"
        ).joinpath("model_smoothbce.pth") as model_path:
            self.filter.load_state_dict(
                torch.load(str(model_path), map_location=self.device)
            )

        # Pool is created lazily in _get_reactions, not here.
        # This avoids issues with fork inheritance and nested ProcessPoolExecutor
        # (e.g. when this model runs inside server.py's worker processes).
        self._num_sub_workers = int(os.getenv("SIMPRETRO_NUM_SUB_WORKERS", "4"))
        self._match_pool = None
        self._match_pool_pid = None  # track which process owns the pool

        # Set global for standalone usage (not inside server worker)
        global _MATCH_MODEL
        _MATCH_MODEL = self

    def get_parameters(self):
        """Return model parameters for optimization."""
        return self.filter.parameters()

    def close(self):
        """Close subprocess pool. Call explicitly to avoid interpreter shutdown issues."""
        if self._match_pool is not None:
            try:
                self._match_pool.close()
                self._match_pool.join()
            except Exception:
                pass
            self._match_pool = None

    def _get_reactions(
        self, inputs: List[Molecule], num_results: int
    ) -> List[Sequence[SingleProductReaction]]:
        """Generate reaction predictions for input molecules.

        Single molecule: run locally (no fork overhead).
        Multiple molecules + main process: use forked subprocess pool.
        Worker process (inside server's ProcessPoolExecutor): always local.
        """
        from functools import partial

        n = len(inputs)
        use_pool = (n > 1 and
                    mp.current_process().name == "MainProcess" and
                    self._num_sub_workers > 0)

        # Lazy pool creation + PID-based reinitialization after fork
        if use_pool:
            current_pid = os.getpid()
            if (self._match_pool is None or
                    self._match_pool_pid != current_pid):
                # Close stale pool from parent process
                if self._match_pool is not None:
                    try:
                        self._match_pool.close()
                        self._match_pool.join()
                    except Exception:
                        pass
                # Create new pool in this process
                self._match_pool = mp.get_context("fork").Pool(self._num_sub_workers)
                _ALL_POOLS.append(self._match_pool)
                self._match_pool_pid = current_pid

        if use_pool:
            match_data = self._match_pool.map(
                partial(_match_single_molecule, num_results=num_results),
                [x.smiles for x in inputs],
            )
        else:
            # Single molecule or worker process: run locally
            match_data = [
                _match_single_molecule(x.smiles, num_results=num_results)
                for x in inputs
            ]

        # Main process: only data conversion, no heavy computation
        return [
            process_raw_smiles_outputs_backwards(
                input=input,
                output_list=list(output[0]),
                metadata_list=[{"probability": p, "score": s, "template": t}
                               for p, s, t in zip(output[1], output[2], output[3])],
            )
            for input, output in zip(inputs, match_data)
        ]
        # 虽然这里使用的变量名叫pred、probability，但其输出与其叫反应发生成功率，不如叫模板价值，神经网络应为排除低价值模板产生的合成路径


# Global registry for atexit cleanup — prevents Pool.__del__ errors on interpreter shutdown.
_ALL_POOLS: List = []


def _cleanup_all_pools():
    for pool in list(_ALL_POOLS):
        try:
            pool.close()
            pool.join()
        except Exception:
            pass


atexit.register(_cleanup_all_pools)