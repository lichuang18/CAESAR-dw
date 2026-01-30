#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化对比脚本：在相同压缩比下比较不同方法的重建效果
支持精确压缩比控制（通过二分搜索）
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 设置路径
sys.path.insert(0, '/home/lch/compress/CAESAR-dw')
sys.path.insert(0, '/home/lch/compress/replicate/CAESAR')

from torch.utils.data import DataLoader


def load_caesar_freq_model(model_path, device='cuda'):
    """加载 CAESAR-Freq 模型"""
    from CAESAR.compressor_freq import CAESARFreq
    compressor = CAESARFreq(
        model_path=model_path,
        use_diffusion=False,
        device=device,
        n_frame=48
    )
    return compressor


def load_caesar_original_model(model_path, use_diffusion=False, n_frame=8, device='cuda'):
    """加载原版 CAESAR 模型"""
    os.chdir('/home/lch/compress/replicate/CAESAR')
    from CAESAR.compressor import CAESAR
    compressor = CAESAR(
        model_path=model_path,
        use_diffusion=use_diffusion,
        device=device,
        n_frame=n_frame,
        interpo_rate=3 if use_diffusion else 1,
        diffusion_steps=32 if use_diffusion else 0
    )
    return compressor


def test_and_get_reconstruction(compressor, dataloader, dataset, eb, method_name):
    """测试并获取重建数据"""
    print(f"  测试 {method_name} (eb={eb:.2e})...")

    compressed_data, compressed_size = compressor.compress(dataloader, eb=eb)
    recons_data = compressor.decompress(compressed_data)

    original_data = dataset.input_data()
    recons_data = dataset.recons_data(recons_data)

    # 计算指标
    nrmse = torch.sqrt(torch.mean((original_data - recons_data) ** 2)) / (torch.max(original_data) - torch.min(original_data))
    cr = np.prod(original_data.shape) * 4 / compressed_size

    print(f"    NRMSE: {nrmse.item():.6f}, CR: {cr:.2f}x")

    return original_data.numpy(), recons_data.numpy(), nrmse.item(), cr, compressed_size


def find_eb_for_target_cr(compressor, dataloader, dataset, target_cr, method_name,
                          eb_min=1e-6, eb_max=1e-1, tolerance=0.02, max_iter=20):
    """
    二分搜索找到目标压缩比对应的 eb

    Args:
        target_cr: 目标压缩比
        tolerance: 允许的误差范围 (如 0.02 表示 ±2%)
        max_iter: 最大迭代次数
    """
    print(f"  为 {method_name} 搜索 target_cr={target_cr}x 对应的 eb...")

    original_data = dataset.input_data()
    original_size = np.prod(original_data.shape) * 4  # float32 = 4 bytes

    best_eb = None
    best_cr = None
    best_diff = float('inf')

    for i in range(max_iter):
        eb_mid = np.sqrt(eb_min * eb_max)  # 几何平均

        compressed_data, compressed_size = compressor.compress(dataloader, eb=eb_mid)
        cr = original_size / compressed_size

        diff = abs(cr - target_cr) / target_cr
        print(f"    iter {i+1}: eb={eb_mid:.2e}, CR={cr:.2f}x, diff={diff*100:.1f}%")

        if diff < best_diff:
            best_diff = diff
            best_eb = eb_mid
            best_cr = cr

        if diff < tolerance:
            print(f"    找到: eb={eb_mid:.2e}, CR={cr:.2f}x")
            return eb_mid, cr

        if cr > target_cr:
            # 压缩比太高，需要更小的 eb（更高精度）
            eb_max = eb_mid
        else:
            # 压缩比太低，需要更大的 eb（更低精度）
            eb_min = eb_mid

    print(f"    最佳结果: eb={best_eb:.2e}, CR={best_cr:.2f}x (diff={best_diff*100:.1f}%)")
    return best_eb, best_cr


def visualize_comparison(results, dataset_name, save_dir, frame_idx=24, zoom_size=64):
    """
    生成可视化对比图，分别保存3张图

    Args:
        results: 各方法的结果列表
        dataset_name: 数据集名称
        save_dir: 保存目录
        frame_idx: 显示的帧索引
        zoom_size: 局部放大区域大小 (越小放大倍数越大，可以用8, 16, 32等)
    """
    n_methods = len(results)

    # 获取原始数据
    original = results[0]['original']

    # 选择一个切片进行可视化
    # 数据格式: (V, S, T, H, W)
    v_idx, s_idx = 0, 0

    # 获取数据范围用于统一colorbar
    vmin = original[v_idx, s_idx, frame_idx].min()
    vmax = original[v_idx, s_idx, frame_idx].max()

    # 选择局部放大区域 (找梯度变化最剧烈的区域)
    frame_data = original[v_idx, s_idx, frame_idx]
    # 计算梯度幅值
    grad_h = np.abs(np.diff(frame_data, axis=0))
    grad_w = np.abs(np.diff(frame_data, axis=1))
    # 用滑动窗口找梯度总和最大的区域
    step = max(1, zoom_size // 8)
    max_grad_sum = 0
    best_h, best_w = 0, 0
    for h in range(0, frame_data.shape[0] - zoom_size, step):
        for w in range(0, frame_data.shape[1] - zoom_size, step):
            grad_sum = (grad_h[h:h+zoom_size-1, w:w+zoom_size].sum() +
                       grad_w[h:h+zoom_size, w:w+zoom_size-1].sum())
            if grad_sum > max_grad_sum:
                max_grad_sum = grad_sum
                best_h, best_w = h, w

    h_start, h_end = best_h, best_h + zoom_size
    w_start, w_end = best_w, best_w + zoom_size
    row_idx = (h_start + h_end) // 2  # 使用放大区域的中间行

    # ========== 图1: 重建对比图 (全图 + 局部放大) ==========
    fig1, axes1 = plt.subplots(1, (n_methods + 1) * 2, figsize=(4 * (n_methods + 1), 4))

    # 原始图
    im = axes1[0].imshow(original[v_idx, s_idx, frame_idx], cmap='viridis', vmin=vmin, vmax=vmax)
    rect = plt.Rectangle((w_start, h_start), w_end - w_start, h_end - h_start,
                         linewidth=2, edgecolor='red', facecolor='none')
    axes1[0].add_patch(rect)
    axes1[0].set_title(f'Original\nFrame {frame_idx}', fontsize=10)
    axes1[0].axis('off')

    axes1[1].imshow(original[v_idx, s_idx, frame_idx, h_start:h_end, w_start:w_end],
                    cmap='viridis', vmin=vmin, vmax=vmax)
    axes1[1].set_title('Zoomed', fontsize=9)
    axes1[1].axis('off')

    for i, res in enumerate(results):
        ax_full = axes1[(i + 1) * 2]
        ax_full.imshow(res['recons'][v_idx, s_idx, frame_idx], cmap='viridis', vmin=vmin, vmax=vmax)
        rect = plt.Rectangle((w_start, h_start), w_end - w_start, h_end - h_start,
                             linewidth=2, edgecolor='red', facecolor='none')
        ax_full.add_patch(rect)
        ax_full.set_title(f"{res['method']}\nCR={res['cr']:.1f}x, NRMSE={res['nrmse']:.2e}", fontsize=9)
        ax_full.axis('off')

        ax_zoom = axes1[(i + 1) * 2 + 1]
        ax_zoom.imshow(res['recons'][v_idx, s_idx, frame_idx, h_start:h_end, w_start:w_end],
                       cmap='viridis', vmin=vmin, vmax=vmax)
        ax_zoom.set_title('Zoomed', fontsize=9)
        ax_zoom.axis('off')

    # colorbar放在最右侧
    cbar_ax1 = fig1.add_axes([0.92, 0.15, 0.015, 0.7])
    fig1.colorbar(im, cax=cbar_ax1)

    fig1.suptitle(f'{dataset_name} - Reconstruction Comparison (CR ≈ 100x)', fontsize=11, fontweight='bold')
    plt.subplots_adjust(left=0.02, right=0.90, wspace=0.08)
    save_path1 = f"{save_dir}/s3d_reconstruction.png"
    fig1.savefig(save_path1, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"  图1已保存: {save_path1}")

    # ========== 图2: 误差图 ==========
    fig2, axes2 = plt.subplots(1, n_methods, figsize=(4 * n_methods, 4))
    if n_methods == 1:
        axes2 = [axes2]

    # 计算误差范围
    max_error = 0
    for res in results:
        error = np.abs(original[v_idx, s_idx, frame_idx] - res['recons'][v_idx, s_idx, frame_idx])
        max_error = max(max_error, error.max())

    for i, res in enumerate(results):
        error = np.abs(original[v_idx, s_idx, frame_idx] - res['recons'][v_idx, s_idx, frame_idx])
        im = axes2[i].imshow(error, cmap='hot', vmin=0, vmax=max_error)
        axes2[i].set_title(f"{res['method']} Error\nMax={error.max():.2e}", fontsize=10)
        axes2[i].axis('off')

    # colorbar放在最右侧
    cbar_ax2 = fig2.add_axes([0.92, 0.15, 0.015, 0.7])
    fig2.colorbar(im, cax=cbar_ax2)

    fig2.suptitle(f'{dataset_name} - Error Maps (Absolute)', fontsize=11, fontweight='bold')
    plt.subplots_adjust(left=0.02, right=0.90, wspace=0.02)
    save_path2 = f"{save_dir}/s3d_error.png"
    fig2.savefig(save_path2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"  图2已保存: {save_path2}")

    # ========== 图3: 剖面对比 (排除尖峰区域) ==========
    fig3 = plt.figure(figsize=(12, 5))

    # 获取原始剖面作为基准
    orig_profile = original[v_idx, s_idx, frame_idx, row_idx, :]

    # 找到尖峰位置
    peak_idx = np.argmax(np.abs(orig_profile - np.median(orig_profile)))
    peak_width = 30
    peak_start = max(0, peak_idx - peak_width)
    peak_end = min(len(orig_profile), peak_idx + peak_width)

    # 创建不包含尖峰的mask
    mask = np.ones(len(orig_profile), dtype=bool)
    mask[peak_start:peak_end] = False

    # 主图：排除尖峰区域，显示细节
    ax_main = fig3.add_subplot(1, 1, 1)

    x = np.arange(len(orig_profile))
    # 只绘制非尖峰区域
    ax_main.plot(x[mask], orig_profile[mask], 'k-', linewidth=2, label='Original')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    linestyles = ['--', '-.', ':', '--']
    for i, res in enumerate(results):
        recons_profile = res['recons'][v_idx, s_idx, frame_idx, row_idx, :]
        ax_main.plot(x[mask], recons_profile[mask],
                    color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)],
                    linewidth=1.5, label=f"{res['method']} (CR={res['cr']:.1f}x)")

    ax_main.set_xlabel('Pixel Position', fontsize=11)
    ax_main.set_ylabel('Value', fontsize=11)
    ax_main.set_title(f'Profile Comparison (Row {row_idx}, Frame {frame_idx})', fontsize=11)
    ax_main.legend(loc='best', fontsize=10)
    ax_main.grid(True, alpha=0.3)

    fig3.suptitle(f'{dataset_name} - Profile Comparison (CR ≈ 100x)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    save_path3 = f"{save_dir}/s3d_profile.png"
    fig3.savefig(save_path3, dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f"  图3已保存: {save_path3}")


def test_s3d_dataset(target_cr=100):
    """测试 S3D 数据集，精确控制压缩比"""
    print("\n" + "=" * 60)
    print(f"S3D 数据集可视化对比 (目标 CR={target_cr}x)")
    print("=" * 60)

    os.chdir('/home/lch/compress/CAESAR-dw')
    from dataset import ScientificDataset

    results = []

    # 1. CAESAR-Freq
    data_arg = {
        "data_path": "/home/lch/compress/CAESAR-dw/data/s3d_full.npz",
        "variable_idx": [0],
        "section_range": [0, 1],
        "frame_range": [0, 48],
        "n_frame": 48,
    }
    dataset = ScientificDataset(data_arg)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    compressor = load_caesar_freq_model(
        "./checkpoints/freq_compressor/best_model_s3d.pth"
    )

    # 二分搜索找到目标压缩比对应的 eb
    eb_freq, cr_freq = find_eb_for_target_cr(
        compressor, dataloader, dataset, target_cr, "CAESAR-Freq",
        eb_min=1e-6, eb_max=1e-2
    )

    original, recons, nrmse, cr, _ = test_and_get_reconstruction(
        compressor, dataloader, dataset, eb=eb_freq, method_name="CAESAR-Freq"
    )
    print(f"  [确认] CAESAR-Freq: eb={eb_freq:.2e}, CR={cr:.2f}x, NRMSE={nrmse:.6f}")
    results.append({
        'method': 'CAESAR-Freq',
        'original': original,
        'recons': recons,
        'nrmse': nrmse,
        'cr': cr
    })

    # 2. CAESAR-D
    data_arg_d = {
        "data_path": "/home/lch/compress/CAESAR-dw/data/s3d_full.npz",
        "variable_idx": [0],
        "section_range": [0, 1],
        "frame_range": [0, 48],
        "n_frame": 16,  # CAESAR-D 使用 16 帧
    }

    os.chdir('/home/lch/compress/replicate/CAESAR')
    from dataset import ScientificDataset as ScientificDatasetOrig

    dataset_d = ScientificDatasetOrig(data_arg_d)
    dataloader_d = DataLoader(dataset_d, batch_size=32, shuffle=False, num_workers=0)

    compressor_d = load_caesar_original_model(
        "./pretrained/caesar_d.pt",
        use_diffusion=True,
        n_frame=16
    )

    # 二分搜索找到目标压缩比对应的 eb
    eb_d, cr_d_search = find_eb_for_target_cr(
        compressor_d, dataloader_d, dataset_d, target_cr, "CAESAR-D",
        eb_min=1e-6, eb_max=1e-2
    )

    original_d, recons_d, nrmse_d, cr_d, _ = test_and_get_reconstruction(
        compressor_d, dataloader_d, dataset_d, eb=eb_d, method_name="CAESAR-D"
    )
    print(f"  [确认] CAESAR-D: eb={eb_d:.2e}, CR={cr_d:.2f}x, NRMSE={nrmse_d:.6f}")
    results.append({
        'method': 'CAESAR-D',
        'original': original_d,
        'recons': recons_d,
        'nrmse': nrmse_d,
        'cr': cr_d
    })

    # 验证压缩比
    print("\n" + "-" * 40)
    print(f"  S3D 压缩比验证:")
    print(f"    CAESAR-Freq: CR={results[0]['cr']:.2f}x (目标: {target_cr}x)")
    print(f"    CAESAR-D:    CR={results[1]['cr']:.2f}x (目标: {target_cr}x)")
    print("-" * 40)

    # 生成可视化
    os.chdir('/home/lch/compress/CAESAR-dw')
    # zoom_size: 局部放大区域大小，越小放大倍数越大 (如 8, 16, 32, 64)
    visualize_comparison(results, "S3D Dataset", "./doc", frame_idx=24, zoom_size=16)

    return results


def test_turbrot_dataset(target_cr=100):
    """测试 Turb-Rot 数据集，精确控制压缩比"""
    print("\n" + "=" * 60)
    print(f"Turb-Rot 数据集可视化对比 (目标 CR={target_cr}x)")
    print("=" * 60)

    results = []

    # 1. CAESAR-Freq
    os.chdir('/home/lch/compress/CAESAR-dw')
    from dataset import ScientificDataset

    data_arg = {
        "data_path": "/home/lch/compress/replicate/CAESAR/data/Turb_Rot_testset.npz",
        "variable_idx": [0],
        "section_range": [0, 1],
        "frame_range": [0, 48],
        "n_frame": 48,
    }
    dataset = ScientificDataset(data_arg)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    compressor = load_caesar_freq_model(
        "./checkpoints/freq_compressor_turbrot/best_model.pth"
    )

    # 二分搜索找到目标压缩比对应的 eb
    eb_freq, cr_freq = find_eb_for_target_cr(
        compressor, dataloader, dataset, target_cr, "CAESAR-Freq",
        eb_min=1e-4, eb_max=1e-1
    )

    original, recons, nrmse, cr, _ = test_and_get_reconstruction(
        compressor, dataloader, dataset, eb=eb_freq, method_name="CAESAR-Freq"
    )
    print(f"  [确认] CAESAR-Freq: eb={eb_freq:.2e}, CR={cr:.2f}x, NRMSE={nrmse:.6f}")
    results.append({
        'method': 'CAESAR-Freq',
        'original': original,
        'recons': recons,
        'nrmse': nrmse,
        'cr': cr
    })

    # 2. CAESAR-D
    data_arg_d = {
        "data_path": "/home/lch/compress/replicate/CAESAR/data/Turb_Rot_testset.npz",
        "variable_idx": [0],
        "section_range": [0, 1],
        "frame_range": [0, 48],
        "n_frame": 16,  # CAESAR-D 使用 16 帧
    }

    os.chdir('/home/lch/compress/replicate/CAESAR')
    from dataset import ScientificDataset as ScientificDatasetOrig

    dataset_d = ScientificDatasetOrig(data_arg_d)
    dataloader_d = DataLoader(dataset_d, batch_size=32, shuffle=False, num_workers=0)

    compressor_d = load_caesar_original_model(
        "./pretrained/caesar_d.pt",
        use_diffusion=True,
        n_frame=16
    )

    # 二分搜索找到目标压缩比对应的 eb
    eb_d, cr_d_search = find_eb_for_target_cr(
        compressor_d, dataloader_d, dataset_d, target_cr, "CAESAR-D",
        eb_min=1e-5, eb_max=1e-2
    )

    original_d, recons_d, nrmse_d, cr_d, _ = test_and_get_reconstruction(
        compressor_d, dataloader_d, dataset_d, eb=eb_d, method_name="CAESAR-D"
    )
    print(f"  [确认] CAESAR-D: eb={eb_d:.2e}, CR={cr_d:.2f}x, NRMSE={nrmse_d:.6f}")
    results.append({
        'method': 'CAESAR-D',
        'original': original_d,
        'recons': recons_d,
        'nrmse': nrmse_d,
        'cr': cr_d
    })

    # 验证压缩比
    print("\n" + "-" * 40)
    print(f"  Turb-Rot 压缩比验证:")
    print(f"    CAESAR-Freq: CR={results[0]['cr']:.2f}x (目标: {target_cr}x)")
    print(f"    CAESAR-D:    CR={results[1]['cr']:.2f}x (目标: {target_cr}x)")
    print("-" * 40)

    # 生成可视化
    os.chdir('/home/lch/compress/CAESAR-dw')
    visualize_comparison(results, "Turb-Rot Dataset", "./doc/turbrot_comparison.png", frame_idx=24)

    return results


def main():
    print("=" * 60)
    print("压缩重建可视化对比 (精确压缩比控制)")
    print("=" * 60)

    # 检查 CUDA
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")

    # 创建输出目录
    os.makedirs('/home/lch/compress/CAESAR-dw/doc', exist_ok=True)

    # 目标压缩比
    target_cr = 100

    # 只测试 S3D
    try:
        test_s3d_dataset(target_cr=target_cr)
    except Exception as e:
        print(f"S3D 测试失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("可视化完成!")
    print("=" * 60)
    print("输出文件:")
    print("  - /home/lch/compress/CAESAR-dw/doc/s3d_reconstruction.png")
    print("  - /home/lch/compress/CAESAR-dw/doc/s3d_error.png")
    print("  - /home/lch/compress/CAESAR-dw/doc/s3d_profile.png")


if __name__ == "__main__":
    main()
