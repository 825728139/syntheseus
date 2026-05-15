"""Inference wrapper for the SimpRetro model.

Template-based retrosynthesis model with neural network filtering.
Combines template matching with heuristics and a fast neural filter.
"""

import importlib.resources
import json
import multiprocessing as mp
import os
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
# Sub-process worker: runs template matching for a single molecule.
# Lives in the forked child; _MATCH_MODEL is set by initializer.
# ---------------------------------------------------------------------------
_MATCH_MODEL: Optional["SimpRetroModel"] = None


def _init_match_worker(model: "SimpRetroModel") -> None:
    global _MATCH_MODEL
    _MATCH_MODEL = model


def _match_single_molecule(smiles: str):
    """Match one molecule — called inside forked subprocess."""
    out = match_all_templates(
        product_smiles=smiles,
        template_library=_MATCH_MODEL.cpp_template_lib,
        config=_MATCH_MODEL.match_config,
    )
    # Convert C++ MatchOutput → plain Python types for IPC
    return (dict(out.results), list(out.valid_template_ids))


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

        # Create forked sub-process pool for parallel template matching.
        # Children inherit TemplateLibrary via copy-on-write (no extra load time).
        self._match_pool = None
        self._use_sub_pool = os.getenv("SIMPRETRO_USE_SUB_POOL", "false").lower() in ("true", "1", "yes")
        self._num_sub_workers = int(os.getenv("SIMPRETRO_NUM_SUB_WORKERS", "4"))

        # Set _MATCH_MODEL for sequential path; fork initializer overwrites in children.
        global _MATCH_MODEL
        _MATCH_MODEL = self

    def get_parameters(self):
        """Return model parameters for optimization."""
        return self.filter.parameters()

    def _get_reactions(
        self, inputs: List[Molecule], num_results: int
    ) -> List[Sequence[SingleProductReaction]]:
        """Generate reaction predictions for input molecules."""
        raw_outputs = []
        threshold = 0.2
        n = len(inputs)

        # Template matching — use forked sub-process pool if available
        if self._use_sub_pool and n > 1:
            if self._match_pool is None:
                self._match_pool = mp.get_context("fork").Pool(
                    self._num_sub_workers,
                    initializer=_init_match_worker,
                    initargs=(self,),
                )
            match_data = self._match_pool.map(
                _match_single_molecule, [x.smiles for x in inputs]
            )
        else:
            match_data = [
                _match_single_molecule(x.smiles) for x in inputs
            ]

        for i, x in enumerate(inputs):
            results, valid_template_id = match_data[i]

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
                for k, v in results.items():
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

            # Sort and select top results
            sorted_results = sorted(
                validated_results.items(),
                key=lambda item: item[1][0] + 0.001 * item[1][-1],
                reverse=True,
            )[:num_results]

            if len(sorted_results) > 0:
                reactants, scores = zip(*sorted_results)
                templates = [t[1] for t in scores]
                scores = [s[0] for s in scores]
                probability = [np.exp(s) for s in scores]
                total = sum(probability)
                if total > 0:
                    probability = [p / total for p in probability]
                else:
                    probability = [1.0 / len(probability)] * len(probability)
                raw_outputs.append((reactants, probability, scores, templates))
            else:
                raw_outputs.append(([], [], [], []))

        # Convert to new format using process_raw_smiles_outputs_backwards
        return [
            process_raw_smiles_outputs_backwards(
                input=input,
                output_list=output[0],
                metadata_list=[{"probability": probability, "score": score, "template": temp_smarts}
                               for probability, score, temp_smarts in zip(output[1], output[2], output[3])],
            )
            for input, output in zip(inputs, raw_outputs)
        ]
        # 虽然这里使用的变量名叫pred、probability，但其输出与其叫反应发生成功率，不如叫模板价值，神经网络应为排除低价值模板产生的合成路径