#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CAESAR-Freq vs 原 CAESAR 对比测试脚本
在 Turb_Rot 数据集上进行公平对比
"""

import os
import sys
import torch
import numpy as np
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from CAESAR.compressor_freq import CAESARFreq
from dataset import ScientificDataset
from torch.utils.data import DataLoader


def test_caesar_freq(eb=1e-4, model_path="./checkpoints/freq_compressor_turbrot/best_model.pth"):
    """测试 CAESAR-Freq"""
    print("=" * 60)
    print(f"测试 CAESAR-Freq (eb={eb})")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Turb_Rot 数据配置
    data_arg = {
        "data_path": "/home/lch/compress/replicate/CAESAR/data/Turb_Rot_testset.npz",
        "variable_idx": [0],
        "section_range": [0, 1],
        "frame_range": [0, 48],
        "n_frame": 16,
        "train": False,
        "test_size": (256, 256),
        "inst_norm": True,
        "norm_type": "mean_range",
    }

    print(f"加载数据: {data_arg['data_path']}")
    dataset = ScientificDataset(data_arg)
    print(f"数据形状: {dataset.data_input.shape}")

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # 加载模型
    print(f"加载模型: {model_path}")
    compressor = CAESARFreq(
        model_path=model_path,
        use_diffusion=False,
        device=device,
        n_frame=16,
        wavelet='haar',
        dct_block_size=8,
        use_dwt=True,
        low_freq_size=16,
    )

    # 压缩
    print("\n开始压缩...")
    start_time = time.time()
    compressed_data, compressed_size = compressor.compress(dataloader, eb=eb)
    compress_time = time.time() - start_time

    # 解压
    print("开始解压...")
    start_time = time.time()
    recons_data = compressor.decompress(compressed_data)
    decompress_time = time.time() - start_time

    # 计算指标
    original_data = dataset.input_data()
    recons_data = dataset.recons_data(recons_data)

    # NRMSE
    mse = torch.mean((original_data - recons_data) ** 2)
    rmse = torch.sqrt(mse)
    data_range = torch.max(original_data) - torch.min(original_data)
    nrmse = rmse / data_range

    # PSNR
    psnr = 20 * torch.log10(data_range / rmse)

    # 压缩比
    original_size = np.prod(original_data.shape) * 4  # float32 = 4 bytes
    cr = original_size / compressed_size

    print(f"  NRMSE: {nrmse.item():.6f}")
    print(f"  PSNR: {psnr.item():.2f} dB")
    print(f"  压缩比 (CR): {cr:.2f}x")
    print(f"  压缩时间: {compress_time:.2f}s")
    print(f"  解压时间: {decompress_time:.2f}s")

    return {
        'eb': eb,
        'nrmse': nrmse.item(),
        'psnr': psnr.item(),
        'cr': cr,
        'compress_time': compress_time,
        'decompress_time': decompress_time,
    }


def main():
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 设备: {torch.cuda.get_device_name(0)}")
    print()

    # 检查文件
    data_path = "/home/lch/compress/replicate/CAESAR/data/Turb_Rot_testset.npz"
    model_path = "./checkpoints/freq_compressor_turbrot/best_model.pth"

    if not os.path.exists(data_path):
        print(f"错误: 数据文件不存在: {data_path}")
        return
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        print("请先运行训练或检查路径")
        return

    print(f"数据文件: {data_path} [OK]")
    print(f"模型文件: {model_path} [OK]")
    print()

    # 测试不同的 eb 值
    eb_values = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
    results = []

    for eb in eb_values:
        try:
            result = test_caesar_freq(eb=eb, model_path=model_path)
            results.append(result)
        except Exception as e:
            print(f"eb={eb} 测试失败: {e}")
            import traceback
            traceback.print_exc()
            results.append({'eb': eb, 'nrmse': None, 'cr': None})
        print()

    # 汇总结果
    print()
    print("=" * 80)
    print("CAESAR-Freq 测试结果汇总 (Turb_Rot 数据集)")
    print("=" * 80)
    print(f"{'eb':<12} {'NRMSE':<15} {'PSNR (dB)':<12} {'CR':<12}")
    print("-" * 80)
    for r in results:
        if r.get('nrmse') is not None:
            print(f"{r['eb']:<12.0e} {r['nrmse']:<15.6f} {r['psnr']:<12.2f} {r['cr']:<12.2f}x")
        else:
            print(f"{r['eb']:<12.0e} {'FAILED':<15}")

    # 原 CAESAR 参考值
    print()
    print("=" * 80)
    print("原 CAESAR 参考值 (Turb_Rot 数据集, eb=1e-4)")
    print("=" * 80)
    print("CAESAR-V: NRMSE=0.000097, CR=31x")
    print("CAESAR-D: NRMSE=0.000100, CR=37x")

    # 找到 eb=1e-4 的结果进行对比
    print()
    print("=" * 80)
    print("对比分析 (eb=1e-4)")
    print("=" * 80)
    for r in results:
        if r.get('eb') == 1e-4 and r.get('nrmse') is not None:
            print(f"CAESAR-Freq: NRMSE={r['nrmse']:.6f}, CR={r['cr']:.2f}x")

            # 计算提升
            caesar_d_cr = 37
            improvement = r['cr'] / caesar_d_cr
            print(f"相比 CAESAR-D (CR=37x): 压缩比提升 {improvement:.2f}x")

            if r['nrmse'] < 0.0001:
                print("精度: 优于 CAESAR-D")
            elif r['nrmse'] < 0.00015:
                print("精度: 与 CAESAR-D 相当")
            else:
                print("精度: 略低于 CAESAR-D")
            break


if __name__ == "__main__":
    main()
