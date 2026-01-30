# CAESAR-Freq 训练指南

## 概述

CAESAR-Freq 使用 DWT+DCT 频率变换替代原 VAE 编码器，实现更低的压缩损失和更好的误差可控性。

## 训练流程

训练分为三个阶段：

```
Phase 1: 频率压缩器训练
    ↓
Phase 2: 扩散模型训练 (Teacher, 32步)
    ↓
Phase 3: 单步蒸馏训练 (Student, 1步)
```

## Phase 1: 训练频率压缩器

```bash
python train_freq_compressor.py \
    --train_data /path/to/train_data.npz \
    --val_data /path/to/val_data.npz \
    --wavelet haar \
    --dct_block_size 8 \
    --use_dwt \
    --latent_channels 32 \
    --low_freq_size 16 \
    --batch_size 8 \
    --epochs 100 \
    --lr 1e-4 \
    --lambda_bpp 0.01 \
    --output_dir ./checkpoints/freq_compressor \
    --device cuda
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--wavelet` | haar | 小波类型 (haar/cdf97) |
| `--dct_block_size` | 8 | DCT块大小 |
| `--use_dwt` | True | 是否使用DWT |
| `--latent_channels` | 32 | 潜在空间通道数 |
| `--low_freq_size` | 16 | 低频区域大小 |
| `--lambda_bpp` | 0.01 | bpp损失权重 |

## Phase 2: 训练扩散模型 (Teacher)

```bash
python train_diffusion.py \
    --train_data /path/to/train_data.npz \
    --val_data /path/to/val_data.npz \
    --freq_compressor_path ./checkpoints/freq_compressor/best_model.pth \
    --wavelet haar \
    --use_dwt \
    --latent_channels 32 \
    --low_freq_size 16 \
    --unet_dim 64 \
    --dim_mults 1 2 4 8 \
    --diffusion_steps 32 \
    --interpo_rate 3 \
    --n_frame 16 \
    --batch_size 4 \
    --epochs 100 \
    --lr 1e-4 \
    --output_dir ./checkpoints/diffusion \
    --device cuda
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--freq_compressor_path` | 必需 | 频率压缩器检查点路径 |
| `--diffusion_steps` | 32 | 扩散步数 |
| `--interpo_rate` | 3 | 帧插值率 (每3帧取1个关键帧) |
| `--n_frame` | 16 | 每组帧数 |

## Phase 3: 单步蒸馏训练 (Student)

```bash
python train_distillation.py \
    --train_data /path/to/train_data.npz \
    --val_data /path/to/val_data.npz \
    --freq_compressor_path ./checkpoints/freq_compressor/best_model.pth \
    --teacher_path ./checkpoints/diffusion/diffusion_best.pth \
    --teacher_steps 32 \
    --wavelet haar \
    --use_dwt \
    --latent_channels 32 \
    --low_freq_size 16 \
    --student_base_channels 64 \
    --student_channel_mults 1 2 4 \
    --student_num_res_blocks 2 \
    --use_perceptual_loss \
    --use_frequency_loss \
    --batch_size 4 \
    --epochs 100 \
    --lr 1e-4 \
    --output_dir ./checkpoints/distillation \
    --device cuda
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--teacher_path` | 必需 | Teacher模型检查点路径 |
| `--teacher_steps` | 32 | Teacher扩散步数 |
| `--student_base_channels` | 64 | Student基础通道数 |
| `--use_perceptual_loss` | True | 使用感知损失 |
| `--use_frequency_loss` | True | 使用频率损失 |

## 使用训练好的模型

### 压缩

```python
from CAESAR import CAESARFreq

# 加载模型
compressor = CAESARFreq(
    model_path='./checkpoints/combined_model.pth',
    use_diffusion=True,
    use_single_step=True,  # 使用单步扩散加速
    device='cuda',
    wavelet='haar',
    use_dwt=True,
)

# 压缩
compressed, total_bytes = compressor.compress(dataloader, eb=1e-3)

# 解压
reconstructed = compressor.decompress(compressed)
```

### 合并检查点

```python
import torch

# 加载各个检查点
freq_ckpt = torch.load('./checkpoints/freq_compressor/best_model.pth')
diff_ckpt = torch.load('./checkpoints/diffusion/diffusion_best.pth')
student_ckpt = torch.load('./checkpoints/distillation/student_best.pth')

# 合并
combined = {
    'freq_compressor': freq_ckpt['model_state_dict'],
    'diffusion': diff_ckpt['diffusion_state_dict'],
    'single_step_diffusion': student_ckpt['student_state_dict'],
}

torch.save(combined, './checkpoints/combined_model.pth')
```

## 数据集格式

数据应为 `.npz` 格式，包含形状为 `[V, S, T, H, W]` 的数组：
- V: 变量数
- S: 切片数
- T: 时间步数
- H: 高度
- W: 宽度

## 硬件要求

| 阶段 | GPU显存 | 建议GPU |
|------|---------|---------|
| Phase 1 | ~8GB | RTX 3090 |
| Phase 2 | ~16GB | RTX 3090 / A100 |
| Phase 3 | ~12GB | RTX 3090 |

## 预期结果

| 指标 | 原CAESAR | CAESAR-Freq |
|------|----------|-------------|
| 压缩比 | 10-50× | 15-75× (+50%) |
| PSNR | baseline | +1-2 dB |
| 解压速度 | ~1000ms | ~55ms (单步) |

## 常见问题

### Q: 训练不收敛？
A: 尝试降低学习率或增加 `lambda_bpp` 权重。

### Q: 显存不足？
A: 减小 `batch_size` 或 `training_size`。

### Q: 单步蒸馏质量下降太多？
A: 尝试渐进式蒸馏 (32→16→8→4→2→1) 或增加训练轮数。
