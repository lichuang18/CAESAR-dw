#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CAESAR-Freq 压缩测试脚本
使用 Turb_Rot_testset 数据集测试频率压缩器，与原 CAESAR 进行对比
"""

import os
import sys
import torch
import numpy as np
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from CAESAR.models.frequency_compressor import FrequencyCompressor
from dataset import ScientificDataset
from torch.utils.data import DataLoader


def test_freq_compressor():
    """测试 CAESAR-Freq 频率压缩器"""
    print("=" * 60)
    print("测试 CAESAR-Freq 频率压缩器")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 数据配置 (与原 CAESAR 测试相同)
    data_arg = {
        "data_path": "/home/lch/compress/replicate/CAESAR/data/Turb_Rot_testset.npz",
        "variable_idx": [0],          # 变量索引
        "section_range": [0, 1],      # section 范围
        "frame_range": [0, 48],       # 帧范围
        "n_frame": 16,                # 帧数
        "train": False,               # 测试模式
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

    model = FrequencyCompressor(
        wavelet='haar',
        dct_block_size=8,
        use_dwt=True,
        latent_channels=32,
        hyper_channels=32,
        low_freq_size=16,
    )

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 压缩和解压
    print("\n开始压缩/解压测试...")

    all_recons = []
    all_original = []
    total_compressed_bits = 0
    total_time = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            x = batch["input"].to(device)
            offset = batch["offset"].to(device)
            scale = batch["scale"].to(device)

            # 压缩
            start_time = time.time()
            compressed = model.compress(x)
            compress_time = time.time() - start_time

            # 解压
            start_time = time.time()
            x_rec = model.decompress(compressed['compressed'], device=device)
            decompress_time = time.time() - start_time

            total_time += compress_time + decompress_time

            # 统计压缩大小
            bpf = compressed['bpf_real'].sum().item()
            total_compressed_bits += bpf

            # 反归一化
            x_denorm = x * scale + offset
            x_rec_denorm = x_rec * scale + offset

            all_original.append(x_denorm.cpu())
            all_recons.append(x_rec_denorm.cpu())

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1} batches...")

    # 合并结果
    original_data = torch.cat(all_original, dim=0)
    recons_data = torch.cat(all_recons, dim=0)

    print(f"\n原始数据形状: {original_data.shape}")
    print(f"重建数据形状: {recons_data.shape}")

    # 计算指标
    mse = torch.mean((original_data - recons_data) ** 2)
    rmse = torch.sqrt(mse)
    data_range = torch.max(original_data) - torch.min(original_data)
    nrmse = rmse / data_range
    psnr = 20 * torch.log10(data_range / rmse)

    # 压缩比
    original_bits = np.prod(original_data.shape) * 32  # float32
    compressed_bytes = total_compressed_bits / 8
    cr = original_bits / total_compressed_bits

    print("-" * 40)
    print(f"CAESAR-Freq 结果:")
    print(f"  NRMSE: {nrmse.item():.6f}")
    print(f"  PSNR: {psnr.item():.2f} dB")
    print(f"  压缩比 (CR): {cr:.2f}x")
    print(f"  压缩后大小: {compressed_bytes:.2f} bytes")
    print(f"  总时间: {total_time:.2f}s")
    print("-" * 40)

    return nrmse.item(), cr, psnr.item()


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

    try:
        nrmse, cr, psnr = test_freq_compressor()

        print("\n")
        print("=" * 60)
        print("测试结果汇总")
        print("=" * 60)
        print(f"CAESAR-Freq: NRMSE={nrmse:.6f}, CR={cr:.2f}x, PSNR={psnr:.2f}dB")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
