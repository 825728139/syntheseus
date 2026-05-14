#!/usr/bin/env python3
"""
simp_tem.py - 单个化学反应模板与产物适合度预测脚本

用法：
    python simp_tem.py --template "[C:1](=[O:2])>>[C:1].[O:2]" --smiles "CC(=O)O"
"""

import torch
import numpy as np

# 从现有模块导入神经网络模型和指纹相关函数
from syntheseus.reaction_prediction.fast_filter.model import (
    Net_orig,
    smarts_to_fingerprint,
    smiles_to_fingerprint,
)


def predict_affinity(model, template, product_smiles, device='cpu'):
    """
    预测模板与产物的适合度

    参数:
        model: 加载好的神经网络模型
        template: SMARTS格式的反应模板
        product_smiles: 产物分子的SMILES字符串
        device: 计算设备 ('cpu' 或 'cuda')

    返回:
        float: 适合度分数 (0-1之间)
    """
    # 计算指纹（使用导入的函数）
    template_fp = smarts_to_fingerprint(template)  # (1, 6144)
    product_fp = smiles_to_fingerprint(product_smiles)  # (1, 2048)

    # 组合得到8192维输入 (6144 + 2048)
    combined_fp = np.concatenate([template_fp, product_fp], axis=1)  # (1, 8192)

    # 转换为PyTorch张量
    data = torch.FloatTensor(combined_fp).to(device)

    # 预测（参考 simpretro.py 第255行）
    with torch.no_grad():
        pred = model(data).squeeze().cpu().numpy()

    return float(pred)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='预测化学反应模板与产物的适合度'
    )
    parser.add_argument(
        '--template', '-t', required=True,
        help='SMARTS格式的反应模板'
    )
    parser.add_argument(
        '--smiles', '-s', required=True,
        help='产物分子的SMILES'
    )
    parser.add_argument(
        '--model', '-m',
        default='syntheseus/reaction_prediction/fast_filter/model_smoothbce.pth',
        help='模型权重文件路径'
    )
    parser.add_argument(
        '--device', '-d', default='cpu', choices=['cpu', 'cuda'],
        help='计算设备'
    )

    args = parser.parse_args()

    # 加载模型
    print(f"加载模型从: {args.model}")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = Net_orig().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # 预测
    print(f"\n反应模板: {args.template}")
    print(f"产物SMILES: {args.smiles}")
    # print(f"计算设备: {device}")

    affinity = predict_affinity(model, args.template, args.smiles, device)

    print(f"适合度分数: {affinity:.4f}")


if __name__ == "__main__":
    # 方式1: 直接运行（在此处修改参数）
    # template: SMARTS格式的反应模板
    # smiles: 产物分子的SMILES
    # model: 模型权重文件路径
    # device: 计算设备 ('cpu' 或 'cuda')
    template = "[O;H0;D1;+0:1]=[C;H0;D3;+0:2]-[c;H0;D3;+0:17]>>C-C-[O;H0;D2;+0:1]-C-C.[c;H1;D2;+0:17].N#[C;H0;D2;+0:2]"
    smiles = "O=C1C(=Cc2ccco2)C(=O)c2ccccc21"
    model_path = "/home/liwenlong/chemTools/retro_syn/syntheseus/syntheseus/reaction_prediction/fast_filter/model_smoothbce.pth"
    device = "cpu"

    # 方式2: 通过命令行参数运行（取消下面一行的注释）
    # import sys; main() if len(sys.argv) > 1 else None

    # 直接运行
    print(f"加载模型从: {model_path}")
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = Net_orig().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"\n反应模板: {template}")
    print(f"产物SMILES: {smiles}")
    # print(f"计算设备: {device}")

    affinity = predict_affinity(model, template, smiles, device)
    print(f"适合度分数: {affinity:.4f}")
