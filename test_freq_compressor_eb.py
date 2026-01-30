#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CAESAR-Freq 压缩测试脚本（支持 eb 参数）
使用 Turb_Rot_testset 数据集测试，与原 CAESAR 进行对比
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


def test_freq_compressor(eb=1e-4):
    """测试 CAESAR-Freq 频率压缩器"""
    print("=" * 60)
    print(f"测试 CAESAR-Freq 频率压缩器 (eb={eb})")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 数据配置 (与原 CAESAR 测试相同)
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
    model_path = "./checkpoints/freq_compressor/best_model.pth"
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
    print(f"压缩完成，压缩后大小: {compressed_size:.2f} bytes, 耗时: {compress_time:.2f}s")

    # 解压
    print("开始解压...")
    start_time = time.time()
    recons_data = compressor.decompress(compressed_data)
    decompress_time = time.time() - start_time
    print(f"解压完成，重建数据形状: {recons_data.shape}, 耗时: {decompress_time:.2f}s")

    # 计算指标
    original_data = dataset.input_data()
    recons_data = dataset.recons_data(recons_data)

    # NRMSE
    nrmse = torch.sqrt(torch.mean((original_data - recons_data) ** 2)) / (torch.max(original_data) - torch.min(original_data))

    # 压缩比
    cr = np.prod(original_data.shape) * 32 / (compressed_size * 8)

    print("-" * 40)
    print(f"CAESAR-Freq 结果 (eb={eb}):")
    print(f"  NRMSE: {nrmse.item():.6f}")
    print(f"  压缩比 (CR): {cr:.2f}x")
    print(f"  压缩时间: {compress_time:.2f}s")
    print(f"  解压时间: {decompress_time:.2f}s")
    print("-" * 40)

    return nrmse.item(), cr


def main():
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 设备: {torch.cuda.get_device_name(0)}")

    # 检查文件
    data_path = "/home/lch/compress/replicate/CAESAR/data/Turb_Rot_testset.npz"
    model_path = "./checkpoints/freq_compressor/best_model.pth"

    if not os.path.exists(data_path):
        print(f"错误: 数据文件不存在: {data_path}")
        return
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        return

    print(f"数据文件: {data_path} [OK]")
    print(f"模型文件: {model_path} [OK]")
    print()

    # 测试不同的 eb 值
    eb_values = [1e-3, 1e-4, 1e-5]
    results = []

    for eb in eb_values:
        try:
            nrmse, cr = test_freq_compressor(eb=eb)
            results.append((eb, nrmse, cr))
        except Exception as e:
            print(f"eb={eb} 测试失败: {e}")
            import traceback
            traceback.print_exc()
            results.append((eb, None, None))
        print("\n")

    # 汇总结果
    print("=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    print(f"{'eb':<12} {'NRMSE':<15} {'CR':<10}")
    print("-" * 40)
    for eb, nrmse, cr in results:
        if nrmse is not None:
            print(f"{eb:<12.0e} {nrmse:<15.6f} {cr:<10.2f}x")
        else:
            print(f"{eb:<12.0e} {'FAILED':<15} {'N/A':<10}")

    print("\n原 CAESAR 参考值:")
    print(f"  CAESAR-V (eb=1e-4): NRMSE=0.000097, CR=31x")


if __name__ == "__main__":
    main()
