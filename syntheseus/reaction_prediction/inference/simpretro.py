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


def CDScore(p_mol, r_mols):     # CDScore(p_mol, r.split("."))，r_mols为smarts列表
    """Calculate complexity difference score between product and reactants."""
    p_atom_count = p_mol.GetNumAtoms()
    n_r_mols = len(r_mols)
    if n_r_mols == 1:
        return 0
    r_atom_count = [
        len([int(num[1:]) for num in re.findall(r":\d+", r_mol) if int(num[1:]) < 900]) # 获取每个反应物中参与反应的原子的数量
        for r_mol in r_mols
    ]
    main_r = r_mols[np.argmax(r_atom_count)]        # 获取参与反应贡献原子最多的反应物
    if len(Chem.MolFromSmiles(main_r).GetAtoms()) >= p_atom_count:  # 若贡献原子最多的反应物中的原子数量>=产物原子数量，则反应没有增加复杂性
        return 0
    MAE = 1 / n_r_mols * sum(
        [abs(p_atom_count / n_r_mols - r_atom_count[i]) for i in range(n_r_mols)]
    )   # 产物原子数量 / 反应物数量 - 反应物中参与反应的原子数量 for 对所有反应物求平均绝对误差
    # 判断反应物均衡状态，反应物大小越接近，MAE约小，复杂性差异约大
    return 1 / (1 + MAE) * p_atom_count


def ASScore(p_mol, r_mol_dict, in_stock):   # ASScore(p_mol, canonical_r_dict标准化smiles字典{smarts：smiles}, self.instock_list药品库列表)
    """Calculate availability score for reactants."""
    p_atom_count = p_mol.GetNumAtoms()
    r_mols = list(r_mol_dict.keys())
    r_atom_count = [
        len([int(num[1:]) for num in re.findall(r":\d+", r_mol) if int(num[1:]) < 900]) # 获取每个反应物中参与反应的原子的数量
        for r_mol in r_mols
    ]
    main_r = r_mols[np.argmax(r_atom_count)]    # 获取参与反应贡献原子最多的反应物
    asscore = 0
    for k, v in r_mol_dict.items():
        if v in in_stock:
            add = len(
                [int(num[1:]) for num in re.findall(r":\d+", k) if int(num[1:]) < 900]  # 获取在库存中的反应物中参与反应的原子的数量
            )
            if len(Chem.MolFromSmiles(main_r).GetAtoms()) < p_atom_count:   # 若贡献原子最多的反应物中的原子数量<=产物原子数量，则评分增加当前反应物分子参与反应的原子数量
                asscore += add
            else:
                asscore += add if add > 2 else 0    # 若 ... > ...，但当前反应物分子参与反应的原子数量>2，则增加评分
        if ("Mg" in v or "Li" in v or "Zn" in v) and v not in in_stock:
            asscore -= 10   # 若Mg、Li、Zn在当前分子中，或当前分子不再库存中，则评分减10
    return asscore


def RDScore(p_mol, r_mols):
    """Calculate ring difference score."""
    p_ring_count = p_mol.GetRingInfo().NumRings()   # 计算产物分子的环数量
    r_rings_s = [r_mol.GetRingInfo().AtomRings() for r_mol in r_mols]   # 计算预测反应物中每个环上的原子的原子索引编号，每个环成一个集合，此r_mol内含rdchiral中得到的反应前后原子映射信息
    r_ring_count = 0
    for r_rings, r_mol in zip(r_rings_s, r_mols):
        for r_ring in r_rings:
            mapnums = [r_mol.GetAtomWithIdx(i).GetAtomMapNum() for i in r_ring] # 获取索引编号对应原子的rdchiral原子映射编号列表
            symbols = [r_mol.GetAtomWithIdx(i).GetSymbol() for i in r_ring] # 获取环上的原子符号列表
            if "B" in symbols or "Si" in symbols:   # 如果环上有B，Si则跳过该环，即该环为差异环，在闭环反应中RDScore值增加。辅助性/临时性的环通常包含B或Si，它们在反应中是重要的。
                continue
            if min(mapnums) < 900:  # 若环上索引原子的映射编号均小于900（均大于900时表示该环不参与反应），则该环为非差异化，在闭环反应中RDScore值降低
                r_ring_count += 1
    if p_ring_count > r_ring_count:
        return 1        # 在有机化学中，含有硼（B路易斯酸催化）或硅（Si临时系绳、官能团掩蔽）的环状结构不仅仅是“乘客”，通常是手性辅助剂、反应中间体或潜能官能团。
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
        print("手动加载私有模板, simpretro.py, line 120")
        self.private_templates = json.load(open(Path("/home/liwenlong/chemTools/retro_syn/syntheseus/syntheseus/cli/private_templates.json")))

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
        print("手动加载库存文件, simpretro.py, line 122, inventory_path:", inventory_path)
        inventory_path = Path("/home/liwenlong/chemTools/retro_syn/syntheseus/emolecules.txt")
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
                    rdscore = RDScore(p_mol, r_mols)    # 计算环形差异得分, RDScore为SimpRetro中定义的函数,p_mol为目标分子的mol对象，r_mols为预测的反应物集合列表，返回值0或1
                    cdscore = CDScore(p_mol, r.split("."))   # 计算产品与反应物之间的复杂度差异得分，p_mol为目标分子的mol对象，r.split为预测的反应物smarts列表。返回值，反应物中参与反应的原子的数量。
                    asscore = ASScore(p_mol, canonical_r_dict, self.instock_list)  # 计算反应物的可用性得分，p_mol为目标分子的mol对象，canonical_r_dict为smarts：smiles字典，instock_list库存可购买分子集合。返回值，依赖反应物中参与反应的原子的数量。
                    mdscore = 1 / len(mapped_curr_results)     # 计算预测反应物个数的倒数,多样性评分
                    score = 1 * (
                        w1 * cdscore
                        + w2 * asscore
                        + w3 * rdscore          # 返回值0或1
                        + w4 * mdscore     # 计算预测反应物个数的倒数
                        + 0.1 * (10 if template_raw in self.private_templates else 0)   # 如果模板在私有模板列表中，则增加5分
                    )
                    # if score > 0:
                    #     print(f"CDScore: {cdscore}, ASScore: {asscore}, RDScore: {rdscore}, MDScore: {mdscore}, Overall Score: {score}")
                    #     print(f"Predicted reactants: {canonical_r}, Template: {template_raw}")
                    #     print(f"num_reactants: {len(mapped_curr_results)}, mapped_curr_results: {mapped_curr_results}")
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
                    pred = self.filter(data).squeeze().cpu().numpy()   # 全连接神经网络的输出，输出该模板-产物对有效的概率值（0-1）
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
                raw_outputs.append((reactants, scores, templates))
            else:
                raw_outputs.append(([], [], []))

        # Convert to new format using process_raw_smiles_outputs_backwards
        return [
            process_raw_smiles_outputs_backwards(
                input=input,
                output_list=output[0],
                metadata_list=[{"probability": score, "template": temp_smarts} for score, temp_smarts in zip(output[1], output[2])],
            )
            for input, output in zip(inputs, raw_outputs)
        ]
        # 虽然这里使用的变量名叫pred、probability，但其输出与其叫反应发生成功率，不如叫模板价值，神经网络应为排除低价值模板产生的合成路径