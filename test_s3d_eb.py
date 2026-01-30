#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CAESAR-Freq S3D 数据集测试脚本
测试不同 eb 值下的压缩效果
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


def test_freq_compressor_s3d(eb=1e-4):
    """测试 CAESAR-Freq 频率压缩器（S3D 数据）"""
    print("=" * 60)
    print(f"测试 CAESAR-Freq (eb={eb})")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # S3D 数据配置
    data_arg = {
        "data_path": "./data/s3d_full.npz",
        "variable_idx": [0],          # 使用第一个变量
        "section_range": [0, 1],      # section 范围
        "frame_range": [0, 48],       # 帧范围 (需要是 n_frame 的倍数)
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
    mse = torch.mean((original_data - recons_data) ** 2)
    rmse = torch.sqrt(mse)
    data_range = torch.max(original_data) - torch.min(original_data)
    nrmse = rmse / data_range

    # PSNR
    psnr = 20 * torch.log10(data_range / rmse)

    # 压缩比
    original_size = np.prod(original_data.shape) * 4  # float32 = 4 bytes
    cr = original_size / compressed_size

    # 比特率
    bpp = (compressed_size * 8) / np.prod(original_data.shape)

    print("-" * 40)
    print(f"CAESAR-Freq 结果 (eb={eb}):")
    print(f"  NRMSE: {nrmse.item():.6f}")
    print(f"  PSNR: {psnr.item():.2f} dB")
    print(f"  压缩比 (CR): {cr:.2f}x")
    print(f"  比特率 (bpp): {bpp:.4f}")
    print(f"  原始大小: {original_size / 1024 / 1024:.2f} MB")
    print(f"  压缩大小: {compressed_size / 1024 / 1024:.4f} MB")
    print(f"  压缩时间: {compress_time:.2f}s")
    print(f"  解压时间: {decompress_time:.2f}s")
    print("-" * 40)

    return {
        'eb': eb,
        'nrmse': nrmse.item(),
        'psnr': psnr.item(),
        'cr': cr,
        'bpp': bpp,
        'compress_time': compress_time,
        'decompress_time': decompress_time,
    }


def main():
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 设备: {torch.cuda.get_device_name(0)}")

    # 检查文件
    data_path = "./data/s3d_full.npz"
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
    eb_values = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
    results = []

    for eb in eb_values:
        try:
            result = test_freq_compressor_s3d(eb=eb)
            results.append(result)
        except Exception as e:
            print(f"eb={eb} 测试失败: {e}")
            import traceback
            traceback.print_exc()
            results.append({'eb': eb, 'nrmse': None, 'cr': None})
        print("\n")

    # 汇总结果
    print("=" * 80)
    print("测试结果汇总")
    print("=" * 80)
    print(f"{'eb':<12} {'NRMSE':<15} {'PSNR (dB)':<12} {'CR':<12} {'bpp':<10}")
    print("-" * 80)
    for r in results:
        if r.get('nrmse') is not None:
            print(f"{r['eb']:<12.0e} {r['nrmse']:<15.6f} {r['psnr']:<12.2f} {r['cr']:<12.2f}x {r['bpp']:<10.4f}")
        else:
            print(f"{r['eb']:<12.0e} {'FAILED':<15}")

    print("\n")
    print("=" * 80)
    print("率失真曲线数据 (可用于绘图)")
    print("=" * 80)
    print("CR\tNRMSE\tPSNR")
    for r in results:
        if r.get('nrmse') is not None:
            print(f"{r['cr']:.2f}\t{r['nrmse']:.6f}\t{r['psnr']:.2f}")


if __name__ == "__main__":
    main()
